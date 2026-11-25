import json
import hashlib
import pandas as pd
import lm_eval

class DumpJSON():
    def __init__(self,
                 obj=None, name=None,
                 read_path='results.json', write_path='results.json'):
        if obj is not None:
            path = obj.anals_results_path+'/'+name+'.json'
            read_path=path
            write_path=path
        
        self.read_path=read_path
        self.write_path=write_path
        
        try:
            with open(self.read_path, 'r') as fp:
                self.results = json.load(fp)
        except:
                self.results = {}

    def count(self):
        return(len(self.results))

    def append(self, x):
        
        json_x = json.dumps(x)
        hash = hashlib.sha1(json_x.encode("UTF-8")).hexdigest()
        hash = hash[:10]
        tmp  = {hash:x}
        self.results.update(**tmp)
    
    def save_to_csv(self):
        self.save()
        self.to_csv()
        
    def save(self):
        with open(self.write_path, 'w') as fp:
            json.dump(self.results, fp)

    def to_csv(self):
        df = pd.DataFrame.from_dict(self.results)
        df = df.transpose()
        filename = self.write_path
        filename = filename.split('.')
        if len(filename)>1:
            filename[-1] = 'csv'

        filename = '.'.join(filename)
        df.to_csv(filename)

def calculate_avg_accuracy(task_names: str, results: dict) -> float:
    n_tasks = len(task_names)
    acc_cumul = sum(
        result.get('acc_norm,none', result['acc,none']) for task, result in results.items() if 'mmlu' not in task
    )

    questions_per_mmlu_task = {
        task_name: lm_eval.tasks.get_task_dict([task_name])[task_name].dataset["test"].num_rows
        for task_name in task_names
        if 'mmlu' in task_name
    }

    if not questions_per_mmlu_task:
        return acc_cumul / n_tasks

    # Calculate average accuracy for mmlu tasks, weighted by number of questions in each task
    acc_mmlu = sum(
        result.get('acc_norm,none', result['acc,none']) * questions_per_mmlu_task[task]
        for task, result in results.items()
        if 'mmlu' in task
    )
    acc_mmlu_avg = acc_mmlu / sum(questions_per_mmlu_task.values())

    return (acc_cumul + acc_mmlu_avg) / (n_tasks - len(questions_per_mmlu_task) + 1)