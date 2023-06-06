from llm import *
from tqdm import tqdm
from multiprocessing import Pool
import os


BATCH_SIZE = 15
SPLIT_SIGN = ['\n', "\n\n"][1]
TERM_TABEL = {
    "prompt": "提示词",
    "something-based": "基于某某的",
}
KEEP_TABLE = ["token",]


def prompt(original: str):
    p = r"""You will translate texts from English academic papers written in LaTeX into Chinese.
You will translate only the text contents, while keep commands such as “~\cite{}” or “~\ref{}” the way they were originally.
You will use Chinese in a fluent and academic style.
You will insert a new line whenever there is a sentence terminating, marked by a“。”, “！”, “？” etc.
You will return the result in code format.
You will translate terms according to following rules:
"""
    for term in TERM_TABEL.keys():
        p += f"“{term}” will be translated to “{TERM_TABEL[term]}”;\n"
    for keep in KEEP_TABLE:
        p += f"“{keep}” will be kept as “{keep}”, without being translated;\n"
    p +=r"""
Here is an example.

Original LaTeX in English:
To design such a planning process, we return to the origins of artificial intelligence (and cognitive science), drawing inspiration from the planning processes explored by Newell, Shaw, and Simon starting in the 1950s~\cite{newell1959report, newell1972human}. Newell and colleagues characterized 
 problem solving~\cite\{newell1959report}
as search through a combinatorial problem space, represented as a tree.  We thus propose the Tree of Thoughts (ToT) framework for general problem solving with language models. As Figure~\ref{fig:schematic} illustrates, while existing methods (detailed below) sample continuous language sequences for problem solving, ToT actively maintains a tree of thoughts, where each {\em thought} is a coherent language sequence that serves as an intermediate step toward problem solving (Table~\ref{tab:overview}).

Translated LaTeX in Chinese:
为了设计这样一个规划过程，我们回到人工智能（和认知科学）的起源，借鉴了Newell、Shaw和Simon在1950年代开始探索的规划过程~\cite{newell1959report, newell1972human}。
Newell和同事将问题求解~\cite{newell1959report}描述为对组合问题空间的搜索，其表示为一棵树。
因此，我们提出了“思维树”（Tree of Thoughts，ToT）框架，用于基于语言模型的通用问题求解。
如图~\ref{fig:schematic} 所示，现有方法（详见下文）通过采样连续的语言序列进行问题求解，而ToT则主动地维护一棵思维之树，其中每个“思维”是一个连贯的语言序列，作为解决问题的中间步骤（表~\ref{tab:overview}）。

Here is the case you will work on.

Original LaTeX in English:
"""
    p += original
    p += r"""

Translated LaTeX in Chinese:
"""
    return p

def split_tex_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
        segments = content.split(SPLIT_SIGN)
        return segments
    
def tl_query(original: str):
    if len(original.strip()) == 0:
        return original
    else:
        session = GPTSession()
        res = session.ask(prompt(original))
        return res

def main():
    file_path = input("Input .tex file path: ")
    file_dir, file_name = os.path.split(file_path)
    file_name, file_ext = os.path.splitext(file_name)
    target_name = file_name + "_tl" + file_ext
    target_path = os.path.join(file_dir, target_name)

    segments = split_tex_file(file_path)
    max_word_count = max(len(segment.split()) for segment in segments)
    print("Reading .tex file done.")
    print(f"Total segments: {len(segments)};")
    print(f"Max segment word count: {max_word_count}, keep this number below 1000 will be safe.")

    batches = [segments[i: i+BATCH_SIZE] for i in range(0,len(segments),BATCH_SIZE)]
        
    tl_segments = []

    for batch in tqdm(batches):
        with Pool(len(batch)) as pool:
            res = pool.map(tl_query, batch)
        res.append("")
        res = SPLIT_SIGN.join(res)
        # print(res)
        with open(target_path, 'a') as file:
            file.write(res)

    
if __name__ == "__main__":
    main()