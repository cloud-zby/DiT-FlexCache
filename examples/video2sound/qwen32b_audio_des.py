import openai
import time
import argparse
import re

def test_vllm_openai_api(client, model_name, prompt, max_tokens=50, temperature=0.7, n=1):
    """
    测试vLLM的OpenAI兼容API
    
    参数:
        base_url: vLLM服务的URL (例如 "http://localhost:8000/v1")
        model_name: 要使用的模型名称
        prompt: 输入的提示文本
        max_tokens: 生成的最大token数
        temperature: 采样温度
        n: 生成多少个完成结果
    """
    
    # print(f"\n测试vLLM OpenAI API (模型: {model_name})")
    # print(f"提示: {prompt}")
    
    try:
        start_time = time.time()
        
        # 调用聊天补全API
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.7,
            n=n,
            seed=0
        )
        
        elapsed_time = time.time() - start_time
        
        # 打印结果
        # print("\n生成结果:")
        # for i, choice in enumerate(response.choices):
        #     print(f"\n结果 {i+1}:")
        #     print(choice.message.content)
        
        # 打印性能信息
        print("\n性能指标:")
        print(f"总耗时: {elapsed_time:.2f}秒")
        if response.usage:
            print(f"提示token数: {response.usage.prompt_tokens}")
            print(f"生成token数: {response.usage.completion_tokens}")
            print(f"总token数: {response.usage.total_tokens}")
            if elapsed_time > 0:
                tokens_per_sec = response.usage.completion_tokens / elapsed_time
                print(f"生成速度: {tokens_per_sec:.2f} tokens/秒")
        return response.choices[0].message.content

    except Exception as e:
        print(f"调用API时出错: {str(e)}")

# video_text="刘亦菲弹钢琴"
# video_text="刘亦菲雨中弹钢琴"
# video_text = "跳水运动"
# video_text = "打乒乓球"
# video_text = "森林燃烧"
# video_text = "瀑布"

video_text = "刘亦菲在大街上游玩"

"""
example:
林语在天桥上做体操 none
林语在广场上演讲 none
林语在演唱会舞台上吹萨克斯 saxophone
林语在小岛沙滩边看火山喷发 volcano eruption
张晨在厨房做饭 cooking sounds
张晨在教室里唱歌 none
张晨在体育馆中跳水 splash water
张晨在音乐厅拉小提琴 violin
张晨在悬崖边上练武术 footsteps, punches
刘亦菲弹钢琴 piano music
刘亦菲雨中弹钢琴 piano
打乒乓球 ping pong hit
瀑布 waterfall flow
帆船在水中航行 water flowing
刘亦菲在大街上游玩 none
"""

content=f"""
根据下列文本识别可能的声音事件，遵循以下规则:
(1) 通过文本描述详细分析场景，分析场景中可能存在的声音事件；
(1) 这些声音事件的类别可能包括乐器声、运动、自然声、环境音、动物声、机械声、日常生活声等；
(2) 根据优先级识别主要的事件，并直接输出声音本身的英文描述，确保为1到3个英文单词；
(3) 当可能存在语音、对话、歌唱、欢呼时，输出“none”；
(4) 如果没有明显的声音事件，输出“none”；
(5) 输出包括（a）思考过程（b）用 {{}} 包裹的英文声音事件描述。
视频描述：{video_text}。
"""

def main():
    parser = argparse.ArgumentParser(description="测试vLLM的OpenAI兼容API")
    parser.add_argument("--base-url", default="http://10.127.16.103:1234/v1", 
                        help="vLLM服务的URL (默认: http://localhost:8000/v1)")
    parser.add_argument("--model", default="/nvfile-heatstorage/model_zoo/huggingface/models--Qwen--Qwen2.5-32B-Instruct/snapshots/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd",
                        help="模型名称 (默认: gpt-3.5-turbo)")
    parser.add_argument("--max-tokens", type=int, default=8192, 
                        help="生成的最大token数 (默认: 8192)")
    parser.add_argument("--temperature", type=float, default=0.7, 
                        help="采样温度 (默认: 0.7)")
    parser.add_argument("--n", type=int, default=1, 
                        help="生成多少个完成结果 (默认: 1)")
    
    args = parser.parse_args()
    
    # 配置OpenAI客户端
    client = openai.OpenAI(
        base_url=args.base_url,
        api_key="no-api-key-required"  # vLLM通常不需要API密钥
    )

    scripts=test_vllm_openai_api(
        client=client,
        model_name=args.model,
        prompt=content,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        n=args.n
    )
    print(scripts)
    pattern = r"\{(.*?)\}"
    match = re.search(pattern, scripts)
    print(match.group(1))

if __name__ == "__main__":
    main()