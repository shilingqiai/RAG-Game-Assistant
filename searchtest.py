from duckduckgo_search import DDGS
import requests

def test_search():
    print("🔍 测试 DDG 搜索...")
    try:
        results = DDGS().text("原神 5.0版本 纳塔 新角色", max_results=2)
        for r in results:
            print(f"标题: {r['title']}\n链接: {r['href']}\n")
            return r['href'] # 返回第一个链接用来测试下一步
    except Exception as e:
        print(f"搜索失败: {e}")
        return None

def test_jina(url):
    print(f"\n📖 测试 Jina 读取网页: {url}")
    try:
        response = requests.get(f"https://r.jina.ai/{url}", timeout=10)
        print(f"状态码: {response.status_code}")
        print(f"截取前200字:\n{response.text[:200]}")
    except Exception as e:
        print(f"读取失败: {e}")

if __name__ == "__main__":
    test_url = test_search()
    if test_url:
        test_jina(test_url)