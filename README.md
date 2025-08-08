# data_to_model
# 一个从数据到模型的项目记录
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