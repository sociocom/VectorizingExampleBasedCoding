
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import argparse
import pandas as pd
from VectorBasedCoding import EntityDictionary, normalize
import numpy as np
import pprint
"""
Command line tool by example based coding.

Command line arguments:

    Path
        input: Input Excel file path
        output: Output Excel file path
    
    Column names
        id: ID column in Excel file
        source: Source column in Excel file
        target: Target column in Excel file
        flag_col: Flag column in Excel file
    
    Flag list
        source_flag: Source flag list for training data. If the flags are more than one, split by comma without space. nan is for the cells missing values.
        target_flag: Target flag list for test data. If the flags are more than one, Split by comma without space. nan is for the cells missing values.
        
    Flag Overwrite
        flag_overwrite: Whether to overwrite the flag column as "D". True or False.

Example usage (with column names):

    python main.py data/DISEASE_test.xlsx data/output.xlsx ID 出現形 正規形 正規形_flag S,A,B,C D,nan False
    
"""

"""Input Example
python main.py data/20240725/DISEASE_test.xlsx data/output.xlsx ID 出現形 正規形 正規形_flag S,A,B,C D,nan False
rye run python main.py data/20240725/DISEASE_test.xlsx data/output.xlsx ID 出現形 正規形 正規形_flag S,A,B,C D,nan False
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""Example usage (with column names):
    python main.py data/DISEASE_test.xlsx data/output.xlsx 行ID 出現形 正規形 正規形_flag S,A,B,C D,nan""",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("input", type=str, help="Input Excel file path")
    parser.add_argument("output", type=str, help="Output Excel file path")

    parser.add_argument("id", type=str, help="ID column in Excel file")
    parser.add_argument("source", type=str, help="Source column in Excel file")
    parser.add_argument("target", type=str, help="Target column in Excel file")
    parser.add_argument("flag_col", type=str, help="Flag column in Excel file")
    
    parser.add_argument("source_flag", type=str,\
        help="Source flag list for training data. If the flags are more than one, split by comma without space.")
    parser.add_argument("target_flag", type=str,\
        help="Target flag list for test data. If the flags are more than one, Split by comma without space.")
    
    parser.add_argument("flag_overwrite", type=str, default=False, help="Whether to overwrite the flag column. True or False.")
    
    args = parser.parse_args()
    
    args.source_flag = args.source_flag.strip("[]").split(",")
    args.target_flag = args.target_flag.strip("[]").split(",")
        
    # コマンドライン引数から値のリストを取得
    args.source_flag = [str(i) if i.lower() != "nan" else np.nan 
                        for i in args.source_flag]
    args.target_flag = [str(i) if i.lower() != "nan" else np.nan 
                        for i in args.target_flag]

    print(f"-Input Information----------------")
    print(f"Target Column: {args.target}")
    print(f"Flag Column: {args.flag_col}")    
    print(f"Source Flag: {args.source_flag}")
    print(f"Target Flag: {args.target_flag}")
    print("----------------------------------")
    
    print("data loading...")
    df_original = pd.read_excel(args.input, index_col=0)
    print("data loading done.")
    
    df = df_original[[args.id, args.source, args.target, args.flag_col]]
    print("-Target Data----------------------------------")
    print(df.head(5))
    print("----------------------------------------------")
    
    train_data = df[df[args.flag_col].isin(args.source_flag)]
    # train_data = train_data[train_data[args.target]!="-1"]
    
    # カラムをstr型に変更
    train_data[args.target] = train_data[args.target].astype(str)
    # train_data = train_data[train_data[args.target]!="[ERR]"]

    train_data[[args.id, args.source, args.target, args.flag_col]].to_csv("train_data.csv")
    
    col_id_words = list(df[df[args.flag_col].isin(args.target_flag)].index)
    id_words = list(df[df[args.flag_col].isin(args.target_flag)][args.id])
    words = df[df[args.flag_col].isin(args.target_flag)][args.source].tolist()
    words = [str(i) for i in words]

    ## Normalize entities
    print("normalizing...")
    normalization_dictionary  = EntityDictionary(\
    'train_data.csv',  args.source,  args.target, 'alabnii/jmedroberta-base-sentencepiece', 'model/model.pth')
    
    normalized, scores  = normalize(words,  normalization_dictionary)
    print("normalizing done.")

    df_results = pd.DataFrame([col_id_words, id_words, words, normalized, scores]).T
    df_results.columns = ["行ID", args.id, args.source, args.target, "score"]
    df_results.set_index("行ID", inplace=True)
    
    if args.flag_overwrite:
        print("Flag Overwrite")
        df_results[args.flag_col] = len(df_results)*["D"]
    # df_results.to_csv("inference.csv")
    
    print("-Modified Data--------------------------------")
    print(df_results.head(5))
    print("----------------------------------------------")
    
    # DataFrameをアップデートする前のコピーを作成
    df_original_copy = df_original.copy()
    
    # マージ
    df_original.update(df_results)
    
    # 実際に変更された要素の数を計算する
    matching_elements = [(i, z, x, y) for i, (z, x, y) in \
        enumerate(zip(df_original[args.source], df_original_copy[args.target], df_original[args.target])) \
        if x != y and not (pd.isnull(x) and pd.isnull(y))]

    print("更新された要素(最初の10件)(行番号，出現形，更新前用語，更新後用語)")
    pprint.pprint(matching_elements[:10])
    print("更新された要素の数:", len(matching_elements))
    pd.DataFrame(matching_elements, columns=["行番号", "出現形", "更新前用語", "更新後用語"]).to_csv("matching_elements.csv")

    ## Write output file
    print("saving...")
    df_original.to_excel(args.output)
    print("saving done.")