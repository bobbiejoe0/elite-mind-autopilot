import os
import requests

from typing import List
from logstream import log

# Largest source clip side to download; the final video is 1080x1920
MAX_SOURCE_DIMENSION = 1920


def search_for_stock_videos(query: str, it: int, min_dur: int) -> List[str]:
    """
    Searches for stock videos using whichever provider has an API key set.
    Pexels is preferred when PEXELS_API_KEY is set, otherwise Pixabay is used.

    Args:
        query (str): The query to search for.
        it (int): The number of results to request.
        min_dur (int): The minimum clip duration in seconds.

    Returns:
        List[str]: A list of stock video URLs.
    """
    pexels_key = os.getenv("PEXELS_API_KEY", "")
    if pexels_key:
        return search_pexels(query, pexels_key, it, min_dur)

    pixabay_key = os.getenv("PIXABAY_API_KEY", "")
    if pixabay_key:
        return search_pixabay(query, pixabay_key, it, min_dur)

    log("[-] No stock video API key set (PEXELS_API_KEY or PIXABAY_API_KEY).", "error")
    return []


def search_pexels(query: str, api_key: str, it: int, min_dur: int) -> List[str]:
    """
    Searches Pexels for stock videos based on a query.

    Args:
        query (str): The query to search for.
        api_key (str): The API key to use.

    Returns:
        List[str]: A list of stock videos.
    """

    # Build headers
    headers = {
        "Authorization": api_key
    }

    # Build URL
    qurl = f"https://api.pexels.com/videos/search?query={query}&per_page={it}"

    # Send the request
    r = requests.get(qurl, headers=headers)

    # Parse the response
    response = r.json()

    # Parse each video
    raw_urls = []
    video_url = []
    video_res = 0
    try:
        # loop through each video in the result
        for i in range(it):
            #check if video has desired minimum duration
            if response["videos"][i]["duration"] < min_dur:
                continue
            raw_urls = response["videos"][i]["video_files"]
            temp_video_url = ""

            # loop through each url to determine the best quality
            for video in raw_urls:
                # Check if video has a valid download link
                if ".com/video-files" in video["link"]:
                    # Only save the URL with the largest resolution (up to 1080p)
                    if max(video["width"] or 0, video["height"] or 0) > MAX_SOURCE_DIMENSION:
                        continue
                    if (video["width"]*video["height"]) > video_res:
                        temp_video_url = video["link"]
                        video_res = video["width"]*video["height"]

            # add the url to the return list if it's not empty
            if temp_video_url != "":
                video_url.append(temp_video_url)

    except Exception as e:
        log("[-] No Videos found.", "error")
        log(str(e), "error")

    # Let user know
    log(f"\t=> \"{query}\" found {len(video_url)} Videos", "info")

    # Return the video url
    return video_url


def search_pixabay(query: str, api_key: str, it: int, min_dur: int) -> List[str]:
    """
    Searches Pixabay for stock videos based on a query.

    Args:
        query (str): The query to search for.
        api_key (str): The API key to use.
        it (int): The number of results to request.
        min_dur (int): The minimum clip duration in seconds.

    Returns:
        List[str]: A list of stock videos.
    """
    params = {
        "key": api_key,
        "q": query,
        # Pixabay only accepts per_page between 3 and 200
        "per_page": max(3, min(it, 200)),
        "safesearch": "true",
    }

    video_url = []
    try:
        r = requests.get("https://pixabay.com/api/videos/", params=params, timeout=30)
        r.raise_for_status()

        for hit in r.json().get("hits", []):
            if hit.get("duration", 0) < min_dur:
                continue

            # Pick the highest resolution rendition (up to 1080p) that has a download link
            best_url = ""
            best_res = 0
            for rendition in hit.get("videos", {}).values():
                url = rendition.get("url", "")
                width = rendition.get("width", 0)
                height = rendition.get("height", 0)
                res = width * height
                # Skip 4K renditions: output is 1080x1920 and decoding many
                # 4K clips at once exhausts memory during rendering
                if max(width, height) > MAX_SOURCE_DIMENSION:
                    continue
                if url and res > best_res:
                    best_url = url
                    best_res = res

            if best_url:
                video_url.append(best_url)

    except Exception as e:
        log("[-] No Videos found.", "error")
        log(str(e), "error")

    log(f"\t=> \"{query}\" found {len(video_url)} Videos", "info")

    return video_url
