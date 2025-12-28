import argparse
import os
import json
import torch
import csv
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM, AutoModelForSeq2SeqLM

# Определение задач
TASKS = {
    "blimp": [
        "anaphor_agreement.json", "argument_structure.json", "binding.json",
        "determiner_noun_agreement.json", "ellipsis.json",
        "filler_gap.json", "irregular_forms.json", "island_effects.json",
        "npi_licensing.json", "quantifiers.json", "subject_verb_agreement.json",
        "case_subjective_pronoun.json", "local_attractor"
    ]
}

# Проверка доступности CUDA
print(torch.cuda.is_available())
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_task_data(task_name, input_str):
    """
    Загружает данные задачи из файлов JSON.
    Предполагается, что файлы находятся в директории tasks/{task_name}/
    """
    task_dir = f"evaluation-pipeline/filter-data_{input_str}/"
    task_file = os.path.join(task_dir, task_name)
    #print(task_file)
    with open(task_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def evaluate_model_on_task(model, tokenizer, task_data, num_fewshot=0):
    """
    Оценивает модель на задаче Zorro (сравнение sentence_good vs sentence_bad).
    
    Args:
        model: предобученная трансформерная модель
        tokenizer: токенизатор для модели
        task_data: список словарей из JSON (каждый — с полями sentence_good, sentence_bad, phenomena)
        num_fewshot: число few-shot примеров (пока не используется)

    Returns:
        accuracy: доля правильно решённых примеров
        per_example_results: список с деталями по каждому примеру
    """
    correct = 0
    total = len(task_data)
    per_example_results = []

    for idx, example in enumerate(task_data):
        sentence_good = example["sentence_good"]
        sentence_bad = example["sentence_bad"]
        phenomena = example["phenomena"]

        # Токенизация обоих предложений
        inputs_good = tokenizer(sentence_good, return_tensors="pt", padding=True, truncation=True).to(device)
        inputs_bad = tokenizer(sentence_bad, return_tensors="pt", padding=True, truncation=True).to(device)

        with torch.no_grad():
            # Вычисляем перплексити (или лог-вероятность) для good и bad
            outputs_good = model(**inputs_good)
            logits_good = outputs_good.logits
            # Перплексити: exp(-sum(log softmax) / seq_len)
            loss_good = torch.nn.functional.cross_entropy(
                logits_good.view(-1, logits_good.size(-1)),
                inputs_good["input_ids"].view(-1),
                reduction="mean"
            )
            ppl_good = torch.exp(loss_good).item()

            outputs_bad = model(**inputs_bad)
            logits_bad = outputs_bad.logits
            loss_bad = torch.nn.functional.cross_entropy(
                logits_bad.view(-1, logits_bad.size(-1)),
                inputs_bad["input_ids"].view(-1),
                reduction="mean"
            )
            ppl_bad = torch.exp(loss_bad).item()

        # Решение модели: где перплексити ниже — там предложение «лучше»
        pred = "good" if ppl_good < ppl_bad else "bad"
        target = "good"  # эталон: sentence_good всегда корректна
        is_correct = pred == target

        if is_correct:
            correct += 1

        per_example_results.append({
            "idx": idx,
            "phenomena": phenomena,
            "sentence_good": sentence_good,
            "sentence_bad": sentence_bad,
            "ppl_good": ppl_good,
            "ppl_bad": ppl_bad,
            "pred": pred,
            "target": target,
            "correct": is_correct
        })

        # Логи для отладки (можно убрать)
        # print(f"Пример {idx}: good={ppl_good:.2f}, bad={ppl_bad:.2f} → pred={pred}, correct={is_correct}")

    accuracy = correct / total if total > 0 else 0.0
    return accuracy, per_example_results


def check_and_create_final_avg_scores_csv(file_name):
    if not os.path.isfile(file_name):
        with open(file_name, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(["Model", "Format", "Final Average Score"])
        print(f"File '{file_name}' created with default header.")
    else:
        print(f"File '{file_name}' already exists.")

def append_row_to_final_avg_scores_csv(file_name, model_path, input_str, final_avg_score):
    final_avg_score_formatted = f"{final_avg_score:.2f}%"
    with open(file_name, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([model_path, input_str, final_avg_score_formatted])
    print(f"Appended row to '{file_name}'.")

def check_and_create_task_results_csv(file_name):
    if not os.path.isfile(file_name):
        with open(file_name, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(["Model", "Task", "Score"])
        print(f"File '{file_name}' created with default header.")
    else:
        print(f"File '{file_name}' already exists.")

def append_row_to_task_results_csv(file_name, model_path, task, task_score):
    with open(file_name, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([model_path, task, f"{task_score:.4f}"])
    print(f"Appended row to {file_name}: {task} with score {task_score:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str,
                        help="Path to huggingface model and tokenizer.")
    parser.add_argument("model_type", type=str, choices=["decoder only", "decoder", "encoder only", "encoder", "encoder-decoder"],
                        help="Language model architecture.")
    parser.add_argument("input_str", type=str)
    parser.add_argument("output_file", type=str)
    parser.add_argument("output_final_avg_score_csv", type=str)
    parser.add_argument("output_task_results_csv", type=str)
    parser.add_argument("output_all_results_json", type=str)
    parser.add_argument("--tasks", "-t", type=str, choices=["blimp", "glue"], default="blimp",
                        help="Tasks on which we evaluate.")
    parser.add_argument("--num_fewshot", "-n", type=int, default=0,
                        help="Number of few-shot examples to show the model for each test example.")
    parser.add_argument("--trust_remote_code", "-r", action="store_true",
                        help="Trust remote code (e.g. from huggingface) when loading model.")
    args = parser.parse_args()

    # Загрузка модели и токенизатора
    model_type_map = {
        "decoder only": AutoModelForCausalLM,
        "decoder": AutoModelForCausalLM,
        "encoder only": AutoModelForMaskedLM,
        "encoder": AutoModelForMaskedLM,
        "encoder-decoder": AutoModelForSeq2SeqLM
    }
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=args.trust_remote_code
    )
    
    model = model_type_map[args.model_type].from_pretrained(
        args.model_path,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    ).to(device)
    
    print(f'Type of evaluation: {args.input_str}')
    print(f'Output file to write results to: {args.output_file}')

    if not os.path.isfile(args.output_file):
        with open(args.output_file, 'w', encoding='utf-8') as file:
            file.write('')
        print(f"File '{args.output_file}' created.")
    else:
        print(f"File '{args.output_file}' already exists.")

    check_and_create_final_avg_scores_csv(args.output_final_avg_score_csv)
    check_and_create_task_results_csv(args.output_task_results_csv)

    tasks = []
    if args.tasks == "all":
        for task_type in TASKS.keys():
            tasks.extend(TASKS[task_type])
    else:
        tasks = TASKS[args.tasks]

    accuracies = {}
    all_per_example_results = []

    for task in tasks:
        if task in TASKS["blimp"]:
          task_title = task.replace(".json", "")  # Удаляем расширение для читаемого названия
            
            # Загрузка данных задачи
          try:
              task_data = load_task_data(task, args.input_str)
          except FileNotFoundError:
              print(f"Файл данных для задачи {task} не найден. Пропускаем...")
              continue
          except json.JSONDecodeError as e:
              print(f"Ошибка чтения JSON для задачи {task}: {e}")
              continue

            # Оценка модели на задаче
          try:
              accuracy, per_example_results = evaluate_model_on_task(
                  model, tokenizer, task_data, args.num_fewshot
              )
          except Exception as e:
              print(f"Ошибка при оценке задачи {task}: {e}")
              continue

          accuracies[task_title] = accuracy
          all_per_example_results.extend(per_example_results)

          print(f"{task_title}:\t{accuracies[task_title] * 100:.2f}%")

            # Сохранение результатов по задаче
          out_path = os.path.join(
                args.model_path,
                f"zeroshot_{args.input_str}",
                task_title,
                "eval_results.json"
            )
          out_dir = os.path.dirname(out_path)
          if not os.path.exists(out_dir):
              os.makedirs(out_dir)

          with open(out_path, 'w', encoding='utf-8') as out_file:
              filtered_task_results = [
                  {key: result[key] for key in ["pred", "target", "correct"] if key in result}
                  for result in per_example_results
              ]
              json.dump({
                    "eval_accuracy": accuracies[task_title],
                    "per_example_results": filtered_task_results
              }, out_file, ensure_ascii=False, indent=2)
        else:
          print(f"Неподдерживаемая задача: {task}. Пропускаем...")
          continue  # Пропускаем неизвестные задачи

    # Сохранение всех результатов по примерам для статистических тестов
    if all_per_example_results:  # Проверяем, есть ли результаты
        per_example_results_file = args.output_all_results_json
        with open(per_example_results_file, 'a', encoding='utf-8') as file:
            filtered_results = [
                {key: result[key] for key in ["pred", "target", "correct"] if key in result}
                for result in all_per_example_results
            ]
            file.write(json.dumps(filtered_results, ensure_ascii=False) + "\n")
    else:
        print("Нет результатов для сохранения в JSONL.")

    # Вывод и сохранение итоговых результатов
    print("\nScores:")
    total_score = 0
    score_count = 0

    with open(args.output_file, 'a', encoding='utf-8') as file:
        file.write(f'Zorro Scores for: {args.model_path} | {args.input_str}\n\n')

        for task in sorted(accuracies.keys()):  # Сортируем для предсказуемого вывода
            score = accuracies[task] * 100
            print(f"{task}:\t{score:.2f}%")
            file.write(f"{task}:\t{score:.2f}%\n")

            append_row_to_task_results_csv(
                args.output_task_results_csv,
                args.model_path,
                task,
                accuracies[task]
            )

            score_count += 1
            total_score += score

        if score_count > 0:
            final_avg_score = total_score / score_count
            print(f"FINAL AVERAGE SCORE: {final_avg_score:.2f}%")
            file.write(f"\nFINAL AVERAGE SCORE: {final_avg_score:.2f}%\n\n\n\n")

            # Добавление итоговой средней оценки в CSV
            append_row_to_final_avg_scores_csv(
                args.output_final_avg_score_csv,
                args.model_path,
                args.input_str,
                final_avg_score
            )
        else:
            print("Не было успешно оценено ни одной задачи.")
            file.write("\nНе было успешно оценено ни одной задачи.\n\n\n\n")

    print("Оценка завершена!")