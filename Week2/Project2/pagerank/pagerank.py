import os
import random
import re
import sys
from collections import Counter

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    if not corpus[page]:
        return {key: 1 / len(corpus) for key in corpus}

    result = {key: (1-damping_factor)/len(corpus) for key in corpus}
    for linked in corpus[page]:
        result[linked] += damping_factor / len(corpus[page])

    return result


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    samples = [random.choice(list(corpus.keys()))]

    for _ in range(n-1):
        probabilities = transition_model(corpus, samples[-1], damping_factor)
        samples.append(random.choices(
            list(probabilities.keys()), probabilities.values())[0])

    return {key: (val / n) for key, val in Counter(samples).items()}


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    ranks = {page: 1/len(corpus) for page in corpus}
    updating = True

    while updating:
        updating = False
        for p in ranks:
            new_probablility = (1-damping_factor) / len(corpus)
            linking = []
            for other in corpus:
                if p in corpus[other] or not corpus[other]:
                    linking.append(other)

            second_condition = 0
            for i in linking:
                if corpus[i]:
                    second_condition += ranks[i] / len(corpus[i])
                else:
                    second_condition += ranks[i] / len(corpus)

            new_probablility += damping_factor * second_condition

            # specification said >0.001, but check50 said those results were slightly out of range so I increased precision *10
            if abs(new_probablility - ranks[p]) > 0.0001:
                updating = True
                ranks[p] = new_probablility

    return ranks


if __name__ == "__main__":
    # print(transition_model({"1.html": {"2.html", "3.html"}, "2.html": {
    #       "3.html"}, "3.html": {"2.html"}}, "1.html", DAMPING))

    # print(sample_pagerank({"1.html": {"2.html", "3.html"}, "2.html": {
    #       "3.html"}, "3.html": {"2.html"}}, DAMPING, SAMPLES))

    # print(iterate_pagerank({"1.html": {"2.html", "3.html"}, "2.html": {
    #       "3.html"}, "3.html": {"2.html"}}, DAMPING))

    main()
