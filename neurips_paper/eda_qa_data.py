import pandas as pd
import ast


def pprint_qa_data(row):
    dlim = "".join(["="] * 50)
    dlim_sub = "".join(["="] * 25)
    p_str = f"{dlim}\nType: {row['evolution_type']}\n{dlim_sub}\n"
    p_str += f"{dlim}\nMetadata: {row['metadata']}\n{dlim_sub}\n"
    p_str += f"Q: {row['question']}\n{dlim_sub}\n"
    for c_id, context in enumerate(ast.literal_eval(row["contexts"])):
        p_str += f"C{c_id+1}: {context}\n"
    p_str += f"{dlim_sub}\n"
    p_str += f"GTA: {row['ground_truth']}\n{dlim}\n"
    return p_str


mhr_qa_data_df = pd.read_csv("./data/mhr_qa_data_5.csv")
print(mhr_qa_data_df.shape)
print(mhr_qa_data_df.columns)

for r_id, row in mhr_qa_data_df.iterrows():
    print(f"ID: {r_id+1}")
    print(pprint_qa_data(row))
