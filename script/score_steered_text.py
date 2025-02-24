import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from utils.logging import log_info
from utils.SteerScore import *
from utils.settings import PARA_DIR
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from prettytable import PrettyTable

def score_steered_text(manners):  
    try:
        # Determine the number of threads to use (you can adjust this)
        num_threads = 50

        file = os.path.join(PARA_DIR, "score_steered_text.csv")
        if not os.path.exists(file):
            steered_text = get_steered_text(manners)

            # Split the DataFrame into chunks
            chunk_size = len(steered_text) // num_threads
            chunks = [steered_text[i:i+chunk_size] for i in range(0, len(steered_text), chunk_size)]

            # Process chunks in parallel
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                future_to_chunk = {executor.submit(process_chunk, chunk): chunk for chunk in chunks}
                
                for future in as_completed(future_to_chunk):
                    chunk_result = future.result()
                    steered_text.update(chunk_result)

                    # Save progress after each chunk is completed
                    steered_text.to_csv(file, index=False, sep=';', encoding='utf-8')
                    log_info(f"Saved progress for a chunk")

            steered_text.to_csv(file, index=False, sep=';', encoding='utf-8')
            log_info(f"Results have been written to {file}")
        else:
            steered_text = pd.read_csv(file, sep=';', encoding='utf-8')
        return steered_text

    except Exception as e:
        log_error(f"An error occurred in score_steered_text: {e}")
        return None
    
def average_scores():
    try:
        # Read the CSV file
        file_path = os.path.join(PARA_DIR, "score_steered_text.csv")
        df = pd.read_csv(file_path, sep=';')
        
        # Calculate average scores for each manner
        # Calculate average scores for each method and coefficient
        # avg_scores = {}
        methods = df['steering_method'].unique().tolist()
        coefs = df['lambda'].unique().tolist()
        
        # Create a DataFrame to store the results
        results_df = pd.DataFrame(columns=['Steering Method', 'Strength', 'Joyful (Avg)', 'Repetitive or chaotic (Avg)'])
        for method in methods:
            # avg_scores[method] = {}
            for coef in coefs:
                # Get all rows for this method and coefficient
                subset = df[(df['steering_method'] == method) & (df['lambda'] == coef)]
                
                # Calculate average scores for joyful and angry
                joyful_avg = subset['joyful'].mean()
                # joyful_sum = subset['joyful'].sum()
                
                # angry_avg = subset['angry'].mean()
                hallucination_avg = subset['hallucination'].mean()
                # hallucination_sum = subset['hallucination'].sum()

                # Append the values to the results DataFrame
                new_row = pd.DataFrame({
                    'Steering Method': [method],
                    'Strength': [coef],
                    'Joyful (Avg)': [joyful_avg],
                    'Hallucination (Avg)': [hallucination_avg]
                })
                results_df = pd.concat([results_df, new_row], ignore_index=True)

                # # Store the results
                # avg_scores[method][coef] = {
                #     'joyful': {
                #         'sum': joyful_sum,
                #         'avg': joyful_avg
                #     },
                #     'hallucination': {
                #         'sum': hallucination_sum,
                #         'avg': hallucination_avg
                #     }
                # }
        
        # log_info(f"Average scores calculated:\n{avg_scores}")
        log_info(results_df)

        # # Create a PrettyTable object
        # table = PrettyTable()
        # table.field_names = ["Steering Method", "Steering Strength", "Joyful (Avg)", "Hallucination (Avg)"]       

        # # Populate the table
        # for method, strengths in avg_scores.items():
        #     for strength, metrics in strengths.items():
        #         joyful_avg = metrics['joyful']['avg']
        #         hallucination_avg = metrics['hallucination']['avg']
        #         table.add_row([method, strength, f"{joyful_avg:.2f}", f"{hallucination_avg:.2f}"])

        # import pdb; pdb.set_trace()
        # # Sort the table by Steering Method and Steering Strength
        # table.sort_key = lambda row: (row[0], row[1])  # Sort by "Steering Method" and "Steering Strength"

        # # Print the table
        # print(table)
        
        results_df.to_csv(os.path.join(PARA_DIR, "average_scores_gen_sentences.csv"), index=False, sep=';', encoding='utf-8')

    except Exception as e:
        log_error(f"An error occurred in average_scores: {e}")
        return None
    
if __name__ == "__main__":
    # define the manner to score
    manners = ["angry"]
    score_steered_text(manners)
    average_scores()