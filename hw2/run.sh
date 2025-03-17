# 刪除所有 good_solution_{i}.py 檔案
find . -name "good_solution_*.py" -type f -delete

# 刪除 best_solution.py 檔案
find . -name "best_solution.py" -type f -delete

find . -name "submission.csv" -type f -delete

python3 main.py > log.txt 2>&1