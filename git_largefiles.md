# Git 大文件问题解决方案

## 问题描述

在向 GitHub 推送代码时遇到文件大小限制错误，GitHub 限制单个文件不能超过 100MB。

## 遇到的大文件

1. **第一个大文件**: `pre_mod/original.tsv.original_backup` (150MB)
2. **第二个大文件**: `ICD10/pre_mod/ICD10.tsv.ICD10_backup` (154MB)

## 错误信息示例

```
remote: error: File pre_mod/original.tsv.original_backup is 150.00 MB; this exceeds GitHub's file size limit of 100.00 MB
remote: error: GH001: Large files detected. You may want to try Git Large File Storage - https://git-lfs.github.com.
To https://github.com/CarterYin/data_to_model.git
 ! [remote rejected] main -> main (pre-receive hook declined)
error: failed to push some refs to 'https://github.com/CarterYin/data_to_model.git'
```

## 解决方案步骤

### 步骤 1: 检查被跟踪的大文件

```powershell
git ls-files | findstr "original.tsv.original_backup"
git ls-files | findstr "ICD10.tsv.ICD10_backup"
```

### 步骤 2: 从 Git 索引中移除大文件

```powershell
# 移除第一个大文件
git rm --cached "pre_mod/original.tsv.original_backup"

# 移除第二个大文件
git rm --cached "ICD10/pre_mod/ICD10.tsv.ICD10_backup"
```

### 步骤 3: 更新 .gitignore 文件

在 `.gitignore` 文件中添加以下规则：

```gitignore
*.tsv
*.original_backup
*.ICD10_backup
backup/
catboost_info/
demand/
_pycache/
__pycache__/
*.txt
*.log
*.csv
*.json
```

### 步骤 4: 提交移除大文件的更改

```powershell
git add .gitignore
git commit -m "Remove large backup file and update .gitignore"
git commit -m "Remove large ICD10 backup file"
git commit -m "Update .gitignore for ICD10 backup files"
```

### 步骤 5: 从 Git 历史中彻底移除大文件

由于大文件已经在 Git 历史中，需要重写历史来彻底移除：

```powershell
# 设置环境变量以抑制警告
$env:FILTER_BRANCH_SQUELCH_WARNING=1

# 从整个历史中移除第一个大文件
git filter-branch --force --index-filter "git rm --cached --ignore-unmatch pre_mod/original.tsv.original_backup" --prune-empty --tag-name-filter cat -- --all

# 从整个历史中移除第二个大文件
git filter-branch --force --index-filter "git rm --cached --ignore-unmatch ICD10/pre_mod/ICD10.tsv.ICD10_backup" --prune-empty --tag-name-filter cat -- --all
```

### 步骤 6: 清理 Git 仓库

```powershell
# 清理过滤分支的备份引用
git for-each-ref --format="%(refname)" refs/original/ | ForEach-Object { git update-ref -d $_ }

# 清理 reflog
git reflog expire --expire=now --all

# 强制垃圾回收
git gc --prune=now --aggressive
```

### 步骤 7: 强制推送到远程仓库

由于重写了历史，需要强制推送：

```powershell
git push --force origin main
```

## 注意事项

1. **强制推送的风险**: `git push --force` 会重写远程仓库的历史，如果有其他协作者，需要通知他们重新克隆仓库。

2. **本地文件保留**: 使用 `git rm --cached` 只是从 Git 跟踪中移除文件，本地文件仍然存在。

3. **预防措施**: 通过 `.gitignore` 文件可以防止将来意外提交大文件。

## 推荐的最佳实践

1. **使用 .gitignore**: 在项目开始时就配置好 `.gitignore` 文件，排除不需要版本控制的文件类型。

2. **Git LFS**: 对于确实需要版本控制的大文件，考虑使用 Git Large File Storage (LFS)。

3. **定期检查**: 在提交前检查文件大小，避免提交过大的文件。

4. **文件分类**: 将数据文件、备份文件等放在专门的目录中，并在 `.gitignore` 中排除。

## 有用的 Git 命令

```powershell
# 检查仓库中的大文件
git rev-list --objects --all | git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' | sed -n 's/^blob //p' | sort --numeric-sort --key=2 | tail -10

# 检查当前 Git 状态
git status

# 查看被跟踪的文件
git ls-files

# 查看 .gitignore 规则是否生效
git check-ignore -v <filename>
```

## 结果

经过以上步骤，成功解决了大文件推送问题：

- 移除了两个超过 100MB 的备份文件
- 更新了 `.gitignore` 规则防止将来的问题
- 成功推送代码到 GitHub
- 仓库状态清洁，没有待提交的更改

## 创建时间

2025年8月19日

---

*此文档记录了解决 Git 大文件推送问题的完整过程，供将来参考使用。*
