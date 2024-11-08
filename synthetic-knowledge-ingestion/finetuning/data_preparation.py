import argparse
import json

import datasets
from datasets import Dataset, DatasetDict

def read_data(data_paths):
    print(f"Reading data from {data_paths}")
    all_data = {}
    for data_path in data_paths:
        with open(data_path, "r", encoding="utf-8") as input_data:
            data = json.load(input_data)
        for k, v in data.items():
            if k in all_data:
                all_data[k].extend(v)
            else:
                all_data[k] = v
    return all_data


def read_dataset(data_paths):
    print(f"Reading data from {data_paths}")
    dataset_list = []
    for data_path in data_paths:
        dataset = datasets.load_from_disk(data_path)
        dataset_list.append(dataset)
    if len(dataset_list) == 1:
        return dataset_list[0]
    combined_dataset = datasets.concatenate_datasets(dataset_list)
    return combined_dataset


def data_split(data_dict, output_path, train_test_ratio=0.2, train_val_ratio=0.1, task="sft", transform=False):
    if not transform:
        final_list = []
        for k, v in data_dict.items():
            for row in v:
                new_row = {
                    "question": row[0],
                    "answer": row[1],
                    "context": k
                }
                final_list.append(new_row)

        dataset = Dataset.from_list(final_list)
        splits = dataset.train_test_split(test_size=train_test_ratio)
        if train_val_ratio != 0:
            train_val_splits = splits['train'].train_test_split(test_size=train_val_ratio)
            train_dataset = train_val_splits['train']
            val_dataset = train_val_splits['test']
            test_dataset = splits['test']
            dataset_dict = DatasetDict({
                'train': train_dataset,
                'val': val_dataset,
                'test': test_dataset})
        else:
            train_dataset = splits['train']
            test_dataset = splits['test']
            val_dataset = None
            dataset_dict = DatasetDict({
                'train': splits['train'],
                'test': splits['test']})
        print(f"Created data: {dataset_dict}")
        dataset_dict.save_to_disk(output_path)
    else:
        dataset_dict = data_dict
        train_dataset = data_dict['train']
        val_dataset = data_dict.get('val')
        test_dataset = data_dict['test']
    # convert to llama-factory format
    train_dataset_ft = transform_dataset(train_dataset, output_path, task)
    test_dataset_ft = transform_dataset(test_dataset, output_path, task)
    if val_dataset:
        val_dataset_ft = transform_dataset(val_dataset, output_path, task)
        dataset_dict_ft = DatasetDict({
            'train': train_dataset_ft,
            'val': val_dataset_ft,
            'test': test_dataset_ft})
    else:
        dataset_dict_ft = DatasetDict({
            'train': train_dataset_ft,
            'test': test_dataset_ft})
    print(f"Created data for fine-tuning")

    dataset_dict_ft.save_to_disk(f"{output_path}_{task}")

    return dataset_dict, dataset_dict_ft


def transform_dataset(dataset, source="", task="sft"):
    _new_dataset = []
    for i in range(len(dataset)):
        cur_row = dataset[i]
        if task == "sft":
            new_row = {
                "messages": [
                    {
                        "role": "user",
                        "content": cur_row["question"]
                    },
                    {
                        "role": "assistant",
                        "content": cur_row["answer"]
                    }
                ],
                "source": source
            }
        elif task == "pt":
            new_row = {"text": f"question: {cur_row['question']}\nanswer: {cur_row['answer']}"}
        else:
            raise NotImplementedError(f"task={task} is not implemented")
        _new_dataset.append(new_row)
    new_dataset = dataset.from_list(_new_dataset)
    return new_dataset


def write_jsonl(data, filename):
    with open(filename, mode='w', encoding='utf-8') as f:
        for item in data["train"]:
            json_record = json.dumps(item)
            f.write(json_record + '\n')


def add_dataset(code_path, ft_dataset_name, task):
    print(f"Add dataset {ft_dataset_name} for task={task} to llama factory at {code_path}")
    if task == "sft":
        dataset_dict = {
            ft_dataset_name: {
                "file_name": f"{ft_dataset_name}.jsonl",
                "formatting": "sharegpt",
                "columns": {
                    "messages": "messages",
                    "source": "source"
                },
                "tags": {
                    "role_tag": "role",
                    "content_tag": "content",
                    "user_tag": "user",
                    "assistant_tag": "assistant"
                }
            }
        }
    elif task == "pt":
        dataset_dict = {
            ft_dataset_name: {
                "file_name": f"{ft_dataset_name}.jsonl",
                "columns": {
                    "prompt": "text"
                }
            }
        }
    else:
        raise NotImplementedError(f"task={task} is not implemented")
    filename = f"{code_path}/data/dataset_info.json"
    with open(filename, 'r', encoding='utf-8') as file:
        data = json.load(file)
    data.update(dataset_dict)
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune and merge weights for Mistral model")

    data_group = parser.add_argument_group('Data Arguments')
    data_group.add_argument("--llama_factory_path", help="Path to llama factory")
    data_group.add_argument("--raw_data_path", nargs='+', type=str, help="Path to raw QA pairs")
    data_group.add_argument("--output_path", help="Path to output data")
    data_group.add_argument("--train_test_ratio", type=float, default=0.2, help="Percentage of data used for test")
    data_group.add_argument("--train_val_ratio", type=float, default=0.1, help="Percentage of data used for val (not used for now)")
    data_group.add_argument("--task", default="sft", help="task type, pt (pretrain) or sft")
    data_group.add_argument("--transform", action='store_true', help="Transform from dataset dict. If provided, it will load the data from disk and rename the columns")

    # parse arguments
    args = parser.parse_args()
    data_args = argparse.Namespace(
    **{k: v for k, v in vars(args).items()
        if k in [action.dest for action in data_group._group_actions]})
    breakpoint()

    dataset_name = f"{data_args.output_path.split('/')[-1]}_{data_args.task}"
    dataset_output_path = f"{data_args.llama_factory_path}/data/{dataset_name}.jsonl"

    print(f"ft_dataset_name={dataset_name}, \nft_dataset_output_path={dataset_output_path}")

    if not data_args.transform:
        raw_data = read_data(data_args.raw_data_path)
    else:
        raw_data = read_dataset(data_args.raw_data_path)
    _, dataset_dict = data_split(raw_data, data_args.output_path, train_test_ratio=data_args.train_test_ratio, train_val_ratio=data_args.train_val_ratio, task=data_args.task, transform=data_args.transform)

    write_jsonl(dataset_dict, dataset_output_path)
    add_dataset(data_args.llama_factory_path, dataset_name, data_args.task)


if __name__ == "__main__":
    main()
