# 创建仓库
## 使用的git指令
### 仓库初始化
- cd 到本地项目根目录
- git init 在本地初始化Git仓库
### .gitignore文件创建
- 新建.gitignore文件，添加忽视的文件和文件夹
```bash
git add .gitignore
git commit -m "Add .gitignore file"
```
- 如果想忽略一个已经被Git跟踪的文件，你需要先从Git暂存区移除它，然后再提交
```bash
git rm --cached file_name
git commit -m "Remove file_name from tracking"
```
```bash
git add .gitignore
git commit -m "Renew .gitignore file"
```
### 将文件添加到Git暂存区
- 如果你想添加所有文件，可以使用：
```bash
git add .
```

- 如果你只想添加某个特定文件，可以使用：

```bash
git add file_name
```

### 提交文件到本地仓库
```bash
git commit -m "Initial commit"
```
### 连接本地仓库和 GitHub 仓库
- 回到你在 GitHub 上创建的仓库页面，找到类似于这样的命令并复制：
```bash
git remote add origin https://github.com/your-username/your-repository-name.git
```
- 在你的终端里运行这条命令。注意把 your-username 和 your-repository-name 替换成你自己的。

### 将本地代码推送到 GitHub
**检查本地分支名：首先，你可以用以下命令查看你本地所有分支：**
```bash
git branch
```
- 这个命令会列出你本地仓库的分支，并在当前分支前加上一个星号 *。
**推送正确的本地分支**
- 我们发现本地的默认分支名是master

- 历史原因：在很长一段时间里，Git 仓库的默认分支通常都是 master。

- GitHub 的新默认：最近几年，GitHub 为了推广更具包容性的术语，将新创建的仓库默认分支名改成了 main。

- 本地与远程不匹配：如果你是在旧版 Git 或旧仓库上初始化项目，你的本地分支很可能是 master，而你新创建的 GitHub 仓库的默认分支是 main，导致两者不匹配。

**解决方法**
如何统一分支名（可选）

如果你想保持本地和远程分支名一致，你可以选择：

1.重命名本地分支：（推荐）
如果你想把本地的 master 分支重命名为 main，可以使用：
```bash
git branch -M main
```

然后就可以使用 
```bash
git push -u origin main
```
来推送了。

2.强制推送：

如果本地分支是 master，你也可以直接把它推送到远程的 main 分支。
```bash
git push -u origin master:main
```
这会将本地的 master 分支内容推送到远程的 main 分支。

-----


# 后期维护遇到的问题
## 1.不小心把产生的大文件add了，导致push无法正常运行

这个问题是因为 Git 的工作机制。即使你**删除了本地文件**，Git 的历史记录中仍然会保留该大文件的信息。当你执行 `git push` 时，Git 会尝试上传整个提交历史，其中包括了那个大文件。因此，尽管你现在看不见它，它依然存在于你本地的版本库历史中，导致远程仓库拒绝你的推送。

-----

### 解决方案：使用 Git Filter-Repo 或 BFG Repo-Cleaner 清理历史记录

要解决这个问题，你必须**从 Git 的历史记录中永久删除该大文件**，而不是仅仅从你的工作目录中删除。Git 官方推荐使用 **`git filter-repo`** 或者 **`BFG Repo-Cleaner`**。`git filter-repo` 是目前官方推荐的工具，它比老的 `git filter-branch` 更快、更安全。

-----

#### 步骤 1: 安装 Git Filter-Repo

首先，你需要安装 `git filter-repo`。

```bash
# 使用 pip 安装
pip install git-filter-repo
```

-----

#### 步骤 2: 清理 Git 历史记录

接下来，进入你的 Git 仓库目录，然后使用 `git filter-repo` 命令来删除大文件。你需要指定要删除的文件的路径。

**重要提示：** 这个操作会**重写你的整个 Git 历史记录**。在执行此操作之前，强烈建议你**备份你的仓库**，以防万一。

```bash
# 假设你的大文件路径是 train_test/all/code/output_age_model/RandomForest_best_pipeline.joblib
git filter-repo --path train_test/all/code/output_age_model/RandomForest_best_pipeline.joblib --invert-paths
```

  - `--path <文件路径>`: 指定要删除的文件的路径。
  - `--invert-paths`: 这个参数告诉 Git 删除指定路径之外的所有文件。如果你只想删除某个特定的文件，而保留其他文件，可以这样使用。

或者，你也可以使用 `--strip-blobs-bigger-than` 参数来删除所有超过某个大小的 blob（Git 中文件数据的存储形式）。

```bash
# 删除所有大于 100MB 的文件
git filter-repo --strip-blobs-bigger-than 100M
```

-----

#### 步骤 3: 强制推送（Force Push）

由于你重写了 Git 历史记录，你的本地分支将与远程分支不兼容。因此，你需要使用 **`--force`** 参数来强制推送你的新历史记录到远程仓库。

**警告：** 强制推送会覆盖远程仓库上的历史记录。如果你在与其他人协作，请务必先与团队成员沟通，否则可能会导致其他人丢失他们的工作。

```bash
git push --force origin main
```

**或者，使用更安全的 `--force-with-lease`**：

```bash
git push --force-with-lease origin main
```

`--force-with-lease` 会检查远程分支是否在你的本地版本基础上有了新的提交。如果没有，它才会强制推送。这可以防止你意外覆盖其他人的工作。

-----

#### 总结

1.  **安装** `git-filter-repo`。
2.  **备份**你的仓库。
3.  使用 `git filter-repo` **从历史记录中删除**大文件。
4.  使用 `git push --force` 或 `git push --force-with-lease` **强制推送**你的更改。

完成这些步骤后，大文件将从你的整个 Git 历史记录中被移除，你的推送操作也就能成功了。


