from pathlib import Path
import requests
# 設定模型檔案路徑
model_path = "./Meta-Llama-3.1-8B-Instruct-Q8_0.gguf"
if not Path(model_path).exists():
    print("下載 LLaMA 3.1 8B 模型中...")
    url = "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf"
    response = requests.get(url, stream=True)
    with open(model_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print("模型下載完成！")
else:
    print("模型已存在！")

import torch
if not torch.cuda.is_available():
    raise Exception('You are not using the GPU runtime. Change it first or you will suffer from the super slow inference speed!')
else:
    print('正在使用GPU')

"""## Prepare the LLM and LLM utility function

By default, we will use the quantized version of LLaMA 3.1 8B. you can get full marks on this homework by using the provided LLM and LLM utility function. You can also try out different LLM models.

In the following code block, we will load the downloaded LLM model weights onto the GPU first.
Then, we implemented the generate_response() function so that you can get the generated response from the LLM model more easily.

You can ignore "llama_new_context_with_model: n_ctx_per_seq (16384) < n_ctx_train (131072) -- the full capacity of the model will not be utilized" warning.
"""

from llama_cpp import Llama

# Load the model onto GPU
llama3 = Llama(
    "./Meta-Llama-3.1-8B-Instruct-Q8_0.gguf",
    verbose=False,
    n_gpu_layers=-1,
    n_ctx=16384,    # This argument is how many tokens the model can take. The longer the better, but it will consume more memory. 16384 is a proper value for a GPU with 16GB VRAM.
)

def generate_response(_model: Llama, _messages: str) -> str:
    '''
    This function will inference the model with given messages.
    '''
    _output = _model.create_chat_completion(
        _messages,
        stop=["<|eot_id|>", "<|end_of_text|>"],
        max_tokens=512,    # This argument is how many tokens the model can generate, you can change it and observe the differences.
        temperature=0,      # This argument is the randomness of the model. 0 means no randomness. You will get the same result with the same input every time. You can try to set it to different values.
        repeat_penalty=2.0,
    )["choices"][0]["message"]["content"]
    return _output

"""## Search Tool

The TA has implemented a search tool for you to search certain keywords using Google Search. You can use this tool to search for the relevant **web pages** for the given question. The search tool can be integrated in the following sections.
"""

from typing import List
from googlesearch import search as _search
from bs4 import BeautifulSoup
from charset_normalizer import detect
import asyncio
from requests_html import AsyncHTMLSession
import urllib3
urllib3.disable_warnings()

async def worker(s:AsyncHTMLSession, url:str):
    try:
        header_response = await asyncio.wait_for(s.head(url, verify=False), timeout=10)
        if 'text/html' not in header_response.headers.get('Content-Type', ''):
            return None
        r = await asyncio.wait_for(s.get(url, verify=False), timeout=10)
        return r.text
    except:
        return None

async def get_htmls(urls):
    session = AsyncHTMLSession()
    tasks = (worker(session, url) for url in urls)
    return await asyncio.gather(*tasks)

async def search(keyword: str, n_results: int=3) -> List[str]:
    '''
    This function will search the keyword and return the text content in the first n_results web pages.

    Warning: You may suffer from HTTP 429 errors if you search too many times in a period of time. This is unavoidable and you should take your own risk if you want to try search more results at once.
    The rate limit is not explicitly announced by Google, hence there's not much we can do except for changing the IP or wait until Google unban you (we don't know how long the penalty will last either).
    '''
    keyword = keyword[:100]
    # First, search the keyword and get the results. Also, get 2 times more results in case some of them are invalid.
    results = list(_search(keyword, n_results * 2, lang="zh", unique=True))
    # Then, get the HTML from the results. Also, the helper function will filter out the non-HTML urls.
    results = await get_htmls(results)
    # Filter out the None values.
    results = [x for x in results if x is not None]


    ############################ Original code ############################
    # # Parse the HTML.
    # results = [BeautifulSoup(x, 'html.parser') for x in results]
    # # Get the text from the HTML and remove the spaces. Also, filter out the non-utf-8 encoding.
    # results = [''.join(x.get_text().split()) for x in results if detect(x.encode()).get('encoding') == 'utf-8']
    # # Return the first n results.
    # return results[:n_results]
    
    ############################ New code ############################
    # Parse the HTML.
    parsed_results = []
    for html in results:
        try:
            # 嘗試解析 HTML
            soup = BeautifulSoup(html, 'html.parser')
            parsed_results.append(soup)
        except Exception as e:
            # 忽略解析失敗的內容
            print(f"HTML 解析失敗: {str(e)[:100]}...")
            continue
    
    # Get the text from the HTML and remove the spaces. Also, filter out the non-utf-8 encoding.
    text_results = []
    for soup in parsed_results:
        try:
            text = ''.join(soup.get_text().split())
            # 進一步檢查編碼
            if detect(text.encode()).get('encoding') == 'utf-8':
                text_results.append(text)
        except Exception as e:
            print(f"文字提取失敗: {str(e)[:100]}...")
            continue
    # Return the first n results.
    return text_results[:n_results]

# """## Test the LLM inference pipeline"""

# # You can try out different questions here.
# test_question='請問誰是 Taylor Swift？'

# messages = [
#     {"role": "system", "content": "你是 LLaMA-3.1-8B，是用來回答問題的 AI。使用中文時只會使用繁體中文來回問題。"},    # System prompt
#     {"role": "user", "content": test_question}, # User prompt
# ]

# print(generate_response(llama3, messages))

"""## Agents

The TA has implemented the Agent class for you. You can use this class to create agents that can interact with the LLM model. The Agent class has the following attributes and methods:
- Attributes:
    - role_description: The role of the agent. For example, if you want this agent to be a history expert, you can set the role_description to "You are a history expert. You will only answer questions based on what really happened in the past. Do not generate any answer if you don't have reliable sources.".
    - task_description: The task of the agent. For example, if you want this agent to answer questions only in yes/no, you can set the task_description to "Please answer the following question in yes/no. Explanations are not needed."
    - llm: Just an indicator of the LLM model used by the agent.
- Method:
    - inference: This method takes a message as input and returns the generated response from the LLM model. The message will first be formatted into proper input for the LLM model. (This is where you can set some global instructions like "Please speak in a polite manner" or "Please provide a detailed explanation".) The generated response will be returned as the output.
"""

class LLMAgent():
    def __init__(self, role_description: str, task_description: str, llm: str = "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF"):
        self.role_description = role_description
        self.task_description = task_description
        self.llm = llm

    async def inference(self, message: str) -> str:
        if self.llm == 'bartowski/Meta-Llama-3.1-8B-Instruct-GGUF':
            messages = [
                {"role": "system", "content": f"你是一個智慧代理人，負責以下角色：\n{self.role_description}\n請使用繁體中文回答。"},
                {"role": "user", "content": f"【任務說明】\n{self.task_description}\n\n【使用者問題】\n{message}"},
            ]
            return generate_response(llama3, messages)  # ✅ 確保 `generate_response` 也是 async
        else:
            raise NotImplementedError("目前僅支援 Meta-Llama-3.1-8B-Instruct-GGUF")


"""TODO: Design the role description and task description for each agent."""

# 這個代理可以幫助你過濾問題中的不相關內容，讓問題更加精簡且易於回答。
question_extraction_agent = LLMAgent(
    role_description="你是問題過濾專家，負責從使用者輸入的問題中刪除無關資訊，使問題更加簡潔、直接，方便後續處理。",
    task_description="請檢查以下問題描述，去除冗餘或不必要的背景資訊，只保留與核心問題相關的部分。\n"
                     "請注意：\n"
                     "1. **移除過長的背景敘述，保留最重要的提問內容**。\n"
                     "2. **確保問題仍然保有完整的語義，避免刪除必要資訊**。\n"
                     "3. **如果問題本身已經很精簡，則不需修改**。\n\n"
                     "範例：\n"
                     "原始問題：『我最近在學機器學習，想知道卷積神經網路（CNN）是什麼，還有它為什麼比全連接層好？』\n"
                     "過濾後：『CNN 是什麼？為什麼 CNN 比全連接層好？』\n\n"
                     "原始問題：『臺灣電影《咒》講述一位單親媽媽受邪神詛咒的故事。請問《咒》的邪神名為？』\n"
                     "過濾後：『《咒》的邪神名為？』\n\n"
                     "請處理以下問題："
)

# 這個代理可以幫助你提取最準確的關鍵字，以便搜尋工具獲得更好的搜尋結果。
keyword_extraction_agent = LLMAgent(
    role_description="你是關鍵字提取專家，負責從問題中識別最關鍵的詞彙，使搜尋工具能夠獲得最準確的結果。",
    task_description="請根據以下問題，提取最重要的關鍵字（3-5 個），確保搜尋能夠獲得最準確的資訊。\n"
                     "請遵循以下準則：\n"
                     "1. **避免過長的關鍵字**，例如『臺灣大學進階英文免修申請規定』應拆分為『台灣大學, 進階英文免修, TOEFL』。\n"
                     "2. **避免過於籠統或無關的詞**，例如『多少分』、『最新』這類詞應該被忽略。\n"
                     "3. **保留專有名詞、重要的數字或具體內容**，例如『GeForce RTX 50』。\n\n"
                     "範例：\n"
                     "問題：『台灣大學進階英文免修申請規定中，托福網路測驗 TOEFL iBT 要達到多少分才能申請？』\n"
                     "『台灣大學, 進階英文免修, 托福TOEFL iBT』\n\n"
                     "問題：『藝人大S是在去哪個國家旅遊時因病去世？』\n"
                     "『大S, 國家, 旅遊, 去世』\n\n"
                     "錯誤範例：\n"
                     "問題：『「虎山雄風飛揚」是哪間學校的校歌？』\n"
                     "你不應該回答：『根據問題內容，我提取的關鍵字是：『虎山雄風飛揚, 校歌』我保留了專有名詞「校長」和具體描述學校精神的情況，忽略過於籠統或無法提供資訊用的單語。』\n"
                     "正確回答：『虎山雄風飛揚, 校歌』\n\n"
                     "問題：『20+30=?』\n"
                     "你不應該回答：『20,30』\n"
                     "正確回答：『20+30=?』，因為加號在算式中很重要\n\n"
                     "問題：『Meta 的 Llama-3.2 系列模型中，參數量最小的哪個是多少 Billion？』\n"
                     "你不應該回答：『Llama-3.2, 參數量最小』\n"
                     "正確回答：『Llama-3.2, 參數量最小多少』，因為問題的重點在於數量，因此不可省略「多少」\n\n"
                     "請處理以下問題："
)

# 這個代理是回答問題的核心模組，會根據搜尋結果與問題內容，提供準確的回答。
qa_agent = LLMAgent(
    role_description="你是 LLaMA-3.1-8B，是專門用來回答問題的 AI。使用中文時只會使用繁體中文來回答問題，並確保回答具備正確性與邏輯性。",
    task_description="請根據以下問題，提供準確、詳細且符合邏輯的回答。\n"
                     "請務必遵守以下規則：\n"
                     "1. **先整理 Google 搜尋結果，過濾不相關資訊**，再進行回答。\n"
                     "2. **回答應該直接明瞭，不要補充過多背景資訊**。\n"
                     "3. **如果問題是問『是誰』，請直接回答人名；如果問數字，請直接回應數字；問地點時，請直接回應地名**。\n"
                     "4. **請確保回答簡潔，最多不超過 50 個中文字**。\n"
                     "5. **如果搜尋結果不足，請基於已知知識回答，但不要胡亂編造**。\n\n"
                     "範例：\n"
                     "問題：『最新的輝達顯卡是出到 GeForce RTX 多少系列？』\n"
                     "回答：『GeForce RTX 50 系列』\n\n"
                     "問題：『是誰發現了萬有引力？』\n"
                     "回答：『牛頓』\n\n"
                     "問題：『臺灣電影《咒》的邪神名為？』\n"
                     "回答：『大黑佛母』\n\n"
                     "請處理以下問題：<Question>"
)

"""## RAG pipeline

TODO: Implement the RAG pipeline.

Please refer to the homework description slides for hints.

Also, there might be more heuristics (e.g. classifying the questions based on their lengths, determining if the question need a search or not, reconfirm the answer before returning it to the user......) that are not shown in the flow charts. You can use your creativity to come up with a better solution!

- Naive approach (simple baseline)

    ![](https://www.csie.ntu.edu.tw/~ulin/naive.png)

- Naive RAG approach (medium baseline)

    ![](https://www.csie.ntu.edu.tw/~ulin/naive_rag.png)

- RAG with agents (strong baseline)

    ![](https://www.csie.ntu.edu.tw/~ulin/rag_agent.png)
"""
import csv

async def pipeline(question: str) -> str:
    MAX_TOKENS = 16000
    truncated_question = question[:MAX_TOKENS]

    # Step 1: Extract the core question
    core_question = await question_extraction_agent.inference(truncated_question)
    # print("核心問題：", core_question)

    # Step 2: Extract keywords for searching
    keywords = await keyword_extraction_agent.inference(core_question)
    # print("提取的關鍵字：", keywords)

    # Step 3: Search the web using extracted keywords
    search_results = await search(keywords)
    # print("搜尋結果：", search_results)

    if not search_results:
        return "未找到相關搜尋結果。"

    # Step 4: Combine search results into a single string
    search_text = "\n".join(search_results)[:MAX_TOKENS]  # Ensures it doesn't exceed token limit

    # Step 5: Generate the final answer using both the core question and search results
    final_answer = await qa_agent.inference(core_question + "</Question>。以下為搜尋結果，請過濾不相關資訊再回答問題：<SearchResults>" + search_text + "。</SearchResults>")

    # 將核心問題、關鍵字和搜尋結果寫入 CSV 檔案
    with open('result.csv', 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow([core_question, keywords, search_results, final_answer])

    return final_answer

"""## Answer the questions using your pipeline!

Since Colab has usage limit, you might encounter the disconnections. The following code will save your answer for each question. If you have mounted your Google Drive as instructed, you can just rerun the whole notebook to continue your process.
"""

from pathlib import Path
import os

# 檢查 answer 資料夾是否存在，若不存在則創建它
answer_dir = Path('./answer')
if not answer_dir.exists():
    os.makedirs(answer_dir)

async def main():
    """主程式進入點"""
    # Fill in your student ID first.
    STUDENT_ID = "b10901176"

    STUDENT_ID = STUDENT_ID.lower()
    # 處理 public.txt
    with open('./public.txt', 'r') as input_f:
        questions = input_f.readlines()
        correct_answers = [l.strip().split(',')[1] for l in questions]
        questions = [l.strip().split(',')[0] for l in questions]
        for id, question in enumerate(questions, 1):
            # if Path(f"./{STUDENT_ID}_{id}.txt").exists():
            #     continue
            answer = await pipeline(question)
            answer = answer.replace('\n',' ')
            print(id)
            print("問題：",question)
            print("正解：",correct_answers[id-1])
            print("回答：",answer)
            print("--------------------")
            with open(f'./answer/{STUDENT_ID}_{id}.txt', 'w') as output_f:
                print(answer, file=output_f)

    # 處理 private.txt
    with open('./private.txt', 'r') as input_f:
        questions = input_f.readlines()
        for id, question in enumerate(questions, 31):
            # if Path(f"./{STUDENT_ID}_{id}.txt").exists():
            #     continue
            answer = await pipeline(question)
            answer = answer.replace('\n',' ')
            print(id)
            print("問題：",question)
            print("回答：",answer)
            print("--------------------")
            with open(f'./answer/{STUDENT_ID}_{id}.txt', 'a') as output_f:
                print(answer, file=output_f)

    # 合併結果
    with open(f'./{STUDENT_ID}.txt', 'w') as output_f:
        for id in range(1,91):
            with open(f'./answer/{STUDENT_ID}_{id}.txt', 'r') as input_f:
                answer = input_f.readline().strip()
                print(answer, file=output_f)

# 執行主程式
if __name__ == "__main__":
    asyncio.run(main())