import argparse
import base64
import configparser
import datetime
import io
import os
import re
import sys
import time
# import openai
"""Lightweight arXiv helper: only search and download PDFs.

This module used to duplicate Paper/Reader/summary logic found in `chat_paper.py`.
To avoid duplication, this file now only contains functions to fetch arXiv search
results and download PDFs. `get_arxiv_papers(args)` returns a list of
Paper-like objects by creating `chat_paper.Paper` instances at runtime. This
keeps arXiv responsibilities isolated and prevents code duplication.
"""

import argparse
import datetime
import os
import re
import time
from collections import namedtuple

import requests
import tenacity
from bs4 import BeautifulSoup
import logging
import logging_config as logging_config  # ensure logging is configured (writes to chatpaper.log)


ArxivParams = namedtuple(
    "ArxivParams",
    [
        "query",
        "key_word",
        "page_num",
        "max_results",
        "days",
        "sort",
        "save_image",
        "file_format",
        "language",
    ],
)


def validateTitle(title: str) -> str:
    if not title:
        return "untitled"
    title = title[:80]
    rstr = r"[/\\:*?\"<>|]"
    title = re.sub(rstr, "_", title)
    title = re.sub(r"\s+", " ", title)
    title = re.sub(r"_+", "_", title)
    return title.strip(' _') or "untitled"


def get_url(keyword: str, page: int):
    base_url = "https://arxiv.org/search/?"
    params = {
        "query": keyword,
        "searchtype": "all",
        "abstracts": "show",
        "order": "-announced_date_first",
        "size": 50,
    }
    if page > 0:
        params["start"] = page * 50
    return base_url + requests.compat.urlencode(params)


def get_titles(url: str, days: int = 1):
    titles, links, dates = [], [], []
    resp = requests.get(url)
    soup = BeautifulSoup(resp.text, "html.parser")
    articles = soup.find_all("li", class_="arxiv-result")
    today = datetime.date.today()
    last_days = datetime.timedelta(days=days)
    for article in articles:
        try:
            title = article.find("p", class_="title").text.strip()
            link = article.find("span").find_all("a")[0].get('href')
            date_text = article.find("p", class_="is-size-7").text
            date_text = date_text.split('\n')[0].split("Submitted ")[-1].split("; ")[0]
            date_val = datetime.datetime.strptime(date_text, "%d %B, %Y").date()
            if today - date_val <= last_days:
                titles.append(title)
                links.append(link)
                dates.append(date_val)
        except Exception:
            continue
    return titles, links, dates


def get_all_titles_from_web(keyword: str, page_num: int = 1, days: int = 1):
    title_list, link_list, date_list = [], [], []
    for page in range(page_num):
        url = get_url(keyword, page)
        titles, links, dates = get_titles(url, days)
        if not titles:
            break
        title_list.extend(titles)
        link_list.extend(links)
        date_list.extend(dates)
    return title_list, link_list, date_list


def _download_pdf_to_path(url: str, title: str, query: str):
    root_path = os.path.abspath(os.path.dirname(__file__))
    path = os.path.join(root_path, 'academic Papers', validateTitle(query))
    os.makedirs(path, exist_ok=True)
    base_name = f"{validateTitle(title)[:80]}.pdf"
    filename = os.path.join(path, base_name)
    if os.path.exists(filename):
        idx = 1
        while True:
            candidate = os.path.join(path, f"{validateTitle(title)[:80]}-{idx}.pdf")
            if not os.path.exists(candidate):
                filename = candidate
                break
            idx += 1
    r = requests.get(url)
    with open(filename, 'wb') as f:
        f.write(r.content)
    return filename


@tenacity.retry(wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
                stop=tenacity.stop_after_attempt(5), reraise=True)
def try_download_pdf(url: str, title: str, query: str):
    return _download_pdf_to_path(url, title, query)


def get_arxiv_papers(args):
    """Fetch arXiv search results, download PDFs, and return a list of chat_paper.Paper objects.

    This function performs only arXiv-specific tasks (search + download). It dynamically imports
    `chat_paper.Paper` to create Paper instances so we avoid duplicating Paper/Reader implementations.
    """
    titles, links, dates = get_all_titles_from_web(args.query, page_num=getattr(args, 'page_num', 1), days=getattr(args, 'days', 2))
    paper_list = []
    for i, title in enumerate(titles):
        if i + 1 > getattr(args, 'max_results', len(titles)):
            break
        pdf_url = links[i] + ".pdf"
        try:
            filename = try_download_pdf(pdf_url, title, args.query)
        except Exception as e:
            logging.error("下载失败: %s -> %s", pdf_url, e)
            continue
        try:
            from chat_paper import Paper as CPPaper
            paper = CPPaper(path=filename, url=links[i], title=title)
            paper_list.append(paper)
        except Exception as e:
            logging.warning("无法创建 Paper 对象: %s", e)
            paper_list.append(type('SimplePaper', (), {'path': filename, 'url': links[i], 'title': title}))

    return paper_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, default='traffic flow prediction')
    parser.add_argument("--page_num", type=int, default=2)
    parser.add_argument("--max_results", type=int, default=3)
    parser.add_argument("--days", type=int, default=10)
    # Use argparse.Namespace directly so standalone execution doesn't require all
    # fields from the original ArxivParams namedtuple.
    args = parser.parse_args()
    start = time.time()
    papers = get_arxiv_papers(args)
    logging.info("Downloaded %s papers in %.1fs", len(papers), time.time()-start)
