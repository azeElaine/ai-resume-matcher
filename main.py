import os
import json
import hashlib
import re
import pdfplumber
import dashscope
from flask import Flask, request, jsonify
from flask_cors import CORS
from http import HTTPStatus
import redis

# 初始化 Flask
app = Flask(__name__)
CORS(app)  # 允许跨域，供前端调用

# 配置 (建议放入环境变量)
DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "YOUR_ALIYUN_API_KEY")
REDIS_HOST = os.environ.get("REDIS_HOST", "None")
REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD", "")
dashscope.api_key = DASHSCOPE_API_KEY

# 初始化 Redis (如果配置了)
r_client = None
if REDIS_HOST and REDIS_HOST != "None":
    try:
        r_client = redis.Redis(host=REDIS_HOST, port=6379, password=REDIS_PASSWORD, decode_responses=True)
    except Exception as e:
        print(f"Redis connection failed: {e}")

# --- 辅助函数 ---

def extract_text_from_pdf(file_stream):
    """解析 PDF 并清洗文本"""
    text = ""
    with pdfplumber.open(file_stream) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    # 简单清洗：去除多余空白字符
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_cache_key(text, jd_text=""):
    """生成缓存 Key (基于内容的 Hash)"""
    raw = text + jd_text
    return hashlib.md5(raw.encode('utf-8')).hexdigest()

def call_ai_service(prompt):
    """调用通义千问进行提取或评分"""
    try:
        response = dashscope.Generation.call(
            model=dashscope.Generation.Models.qwen_turbo,
            messages=[{'role': 'user', 'content': prompt}],
            result_format='message'
        )
        if response.status_code == HTTPStatus.OK:
            return response.output.choices[0].message.content
        else:
            print(f"AI Error: {response}")
            return None
    except Exception as e:
        print(f"AI Exception: {e}")
        return None

# --- 路由定义 ---

@app.route('/analyze', methods=['POST'])
def analyze_resume():
    """
    接口：上传简历 + (可选) JD
    功能：解析 PDF -> AI 提取信息 -> AI 匹配打分 -> 缓存结果
    """
    # 1. 获取上传的文件
    if 'resume' not in request.files:
        return jsonify({"error": "No resume file provided"}), 400
    
    file = request.files['resume']
    jd_text = request.form.get('jd_text', '') # 获取岗位描述
    
    # 2. 解析 PDF 文本
    try:
        resume_text = extract_text_from_pdf(file)
    except Exception as e:
        return jsonify({"error": f"Failed to parse PDF: {str(e)}"}), 500

    # 3. 检查缓存 (加分项)
    cache_key = get_cache_key(resume_text, jd_text)
    if r_client:
        cached_data = r_client.get(cache_key)
        if cached_data:
            return jsonify(json.loads(cached_data))

    # 4. AI 提取关键信息 (Prompt Engineering)
    extract_prompt = f"""
    请从以下简历文本中提取关键信息。
    简历内容: {resume_text[:2500]}... (截断以防过长)
    
    要求输出为严格的 JSON 格式，不包含 Markdown 标记。包含以下字段：
    - name (姓名)
    - phone (电话)
    - email (邮箱)
    - education (学历简述)
    - experience_years (工作年限，数字)
    - skills (技能列表)
    """
    
    extracted_info_str = call_ai_service(extract_prompt)
    
    # 尝试解析 JSON
    try:
        # 清理可能存在的 Markdown ```json ... ```
        clean_json_str = extracted_info_str.replace("```json", "").replace("```", "")
        extracted_info = json.loads(clean_json_str)
    except:
        extracted_info = {"raw_text": extracted_info_str}

    # 5. AI 匹配评分 (如果有 JD)
    match_result = {}
    if jd_text:
        match_prompt = f"""
        我是招聘官。请对比简历和岗位描述。
        简历内容: {resume_text[:2000]}
        岗位描述: {jd_text[:1000]}
        
        请给出：
        1. score (0-100的整数评分)
        2. analysis (简短的匹配分析，不超过100字)
        
        输出为严格的 JSON 格式。
        """
        match_res_str = call_ai_service(match_prompt)
        try:
            clean_match_str = match_res_str.replace("```json", "").replace("```", "")
            match_result = json.loads(clean_match_str)
        except:
            match_result = {"score": 0, "analysis": "AI Parsing Error"}

    # 6. 构造最终响应
    response_data = {
        "resume_text_preview": resume_text[:200],
        "extracted_info": extracted_info,
        "match_result": match_result
    }

    # 7. 写入缓存
    if r_client:
        r_client.setex(cache_key, 3600, json.dumps(response_data)) # 缓存1小时

    return jsonify(response_data)

@app.route('/', methods=['GET'])
def health_check():
    return "Resume Service is Running!"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9000)