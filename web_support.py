from bs4 import BeautifulSoup
import requests as rq
import selenium as sel
from googlesearch import search


class WebSupport:
    def __init__(self):
        pass

    def google_search(self, query: str, num_results: int = 10):
        results = []
        for url in search(query, num_results=num_results):
            results.append(url)
        return results
    
    def fetch_page_content(self, url: str):
        response = rq.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            return soup.get_text()
        else:
            return None
    
        
if __name__ == "__main__":
    web = WebSupport()
    search_results = web.google_search("Artificial Intelligence", num_results=5)
    for result in search_results:
        print(result)