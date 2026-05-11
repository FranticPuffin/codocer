import requests
import datetime
import json
import os

# 从环境变量中读取名为 GITHUB_TOKEN 的值
# 如果环境里没设置，这里会返回 None
github_token = os.getenv("GITHUB_TOKEN")
def get_top_repos(language, limit=50, days=180):
    # 计算半年前的日期 (ISO 8601 格式)
    date_since = (datetime.datetime.now() - datetime.timedelta(days=days)).strftime('%Y-%m-%d')
    
    # 构建搜索查询：语言 + 创建日期
    # q=language:C+created:>2023-11-01
    query = f"language:{language} created:>{date_since}"
    url = f"https://api.github.com/search/repositories?q={query}&sort=stars&order=desc&per_page={limit}"
    
    # 如果你有 GitHub Token，请填入下面的 headers
    headers = {
    "Accept": "application/vnd.github.v3+json",
    "Authorization": f"token {github_token}" if github_token else ""
}

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        return data.get('items', [])
    except requests.exceptions.RequestException as e:
        print(f"请求 {language} 仓库时出错: {e}")
        return []

def main():
    languages = ['C', 'JavaScript']
    results = {}

    for lang in languages:
        print(f"正在获取过去半年内星数最高的 {lang} 项目...")
        repos = get_top_repos(lang)
        results[lang] = []
        
        for idx, repo in enumerate(repos, 1):
            repo_info = {
                "rank": idx,
                "name": repo['full_name'],
                "stars": repo['stargazers_count'],
                "url": repo['html_url'],
                "description": repo['description']
            }
            results[lang].append(repo_info)
            print(f"{idx}. {repo['full_name']} ({repo['stargazers_count']} stars)")
        print("-" * 30)

    # 将结果保存为 JSON 文件
    with open('github_top_repos.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print("数据已保存至 github_top_repos.json")

if __name__ == "__main__":
    main()