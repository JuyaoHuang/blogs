---
title: Linux指令手册
author: Alen
published: 2025-10-10
description: "linux指令速查手册"
first_level_category: "编程"
second_level_category: "linux"
tags: ['linux']
draft: false
---


# 超全Linux指令参考手册 

## 1. 文件与目录管理 (核心操作)

这是与文件系统交互的基础。

### `ls` (List)
列出目录内容。
- **-l**: 使用长格式列出，显示详细信息。
- **-a**: 列出所有文件，包括以`.`开头的隐藏文件。
- **-h**: 与 `-l` 配合，以人类可读的格式显示大小 (KB, MB)。
- **-R**: 递归列出所有子目录的内容。
- **-t**: 按修改时间排序，最新的在前。
- **示例**: `ls -alh` (最常用的组合之一，显示所有文件的详细信息)。

### `cd` (Change Directory)
切换目录。
- **`cd <目录路径>`**: 切换到指定目录 (e.g., `cd /var/log`)。
- **`cd ..`**: 切换到上一级目录。
- **`cd ~`** 或 **`cd`**: 切换到当前用户的主目录。
- **`cd -`**: 切换到上一次所在的目录。
- **`cd /`**: 切换到根目录。

### `pwd` (Print Working Directory)
显示当前所在的完整目录路径。

### `mkdir` (Make Directory)
创建新目录。
- **-p**: 递归创建。如果路径中的父目录不存在，则一并创建。
- **示例**: `mkdir my_new_project`
- **示例**: `mkdir -p deep/nested/directory`

### `rmdir` (Remove Directory)
删除一个空的目录。

### `rm` (Remove)
删除文件或目录。**此命令极其危险，请谨慎使用！**
- **-r**: 递归删除整个目录及其所有内容。
- **-f**: 强制删除，不进行任何提示。
- **-i**: 交互式删除，在删除前进行询问。
- **示例 (删除文件)**: `rm temp_file.log`
- **示例 (删除目录)**: `rm -r old_project`
- **示例 (终极杀器)**: `rm -rf /path/to/danger_zone` (强制、递归删除，无提示，通常无法恢复)。

### `cp` (Copy)
复制文件或目录。
- **-r**: 递归复制整个目录。
- **-i**: 覆盖前询问。
- **-v**: 显示复制过程。
- **示例 (文件)**: `cp config.yaml config.yaml.bak`
- **示例 (目录)**: `cp -r project_v1/ project_v2/`

### `mv` (Move)
移动或重命名文件/目录。
- **示例 (重命名)**: `mv old_name.txt new_name.txt`
- **示例 (移动)**: `mv my_file.txt ../backup/`

### `touch`
创建一个新的空文件，或更新已存在文件的时间戳。
- **示例**: `touch main.c`

### `ln` (Link)
创建链接文件。
- **-s**: 创建一个软链接（符号链接，类似于Windows的快捷方式）。
- **示例**: `ln -s /var/www/my_app/config.ini /etc/my_app_config.ini`

---

## 2. 文件内容查看与处理

### `cat` (Concatenate)
```
我使用手机登录的验证格式是:
EAP方法: PEAP
阶段二身份验证: MSCHAPV2
CA证书: 无
身份: 2023211684
密码: 123456(保密)
代理: 无
IP设置: DHCP
我确定输入的学号和密码是正确的
```

```
ctrl_interface=DIR=/var/run/wpa_supplicant GROUP=netdev
update_config=1
country=CN

network={
    ssid="BUPT-mobile"
    scan_ssid=1
    key_mgmt=WPA-EAP
    eap=PEAP
    identity="2023211684"
    password="JUyao0819."
    phase2="auth=MSCHAPV2"
}
```

 10.29.226.39

IPv6：2001:da8:215:3c02:f868:7743:2c94:89fd

一次性显示整个文件的内容。适合小文件。

- **-n**: 显示行号。
- **示例**: `cat /etc/os-release`

### `less`
分页查看文件内容，功能强大，可前后翻页和搜索。
- **操作**: `空格键/f` (下翻页), `b` (上翻页), `/关键词` (搜索), `q` (退出)。
- **示例**: `less large_log_file.log`

### `more`
类似`less`的分页查看器，功能较少。

### `head`
显示文件的前几行（默认10行）。
- **-n <数字>**: 显示指定的行数。
- **示例**: `head -n 50 main.c`

### `tail`
显示文件的后几行（默认10行）。
- **-n <数字>**: 显示指定的行数。
- **-f**: **(服务器必备)** 持续跟踪文件末尾新增的内容，常用于实时查看日志。按`Ctrl+C`停止。
- **示例**: `tail -f /var/log/nginx/access.log`

### `wc` (Word Count)
统计文件的行数、单词数、字符数。
- **-l**: 只统计行数。
- **-w**: 只统计单词数。
- **-c**: 只统计字节数。
- **示例**: `wc -l my_script.py`

### `grep` (Global Regular Expression Print)
在文件中搜索包含指定模式的行。
- **-r**: 递归搜索目录。
- **-i**: 忽略大小写。
- **-n**: 显示匹配行的行号。
- **-v**: 显示不匹配的行。
- **示例**: `grep -rin "error" /var/log/` (递归、忽略大小写、带行号地在日志目录中搜索"error")。
- **与管道结合**: `ps aux | grep "nginx"` (查找nginx相关的进程)。

---

## 3. 文件搜索与查找

### `find`
在指定路径下按多种条件查找文件。
- **-name "<文件名模式>"`**: 按名称查找（区分大小写）。
- **-iname "<文件名模式>"`**: 按名称查找（不区分大小写）。
- **-type f**: 只查找文件。
- **-type d**: 只查找目录。
- **-size +<大小>`**: 查找大于指定大小的文件 (e.g., `+100M`)。
- **示例**: `find /home/user -name "*.py" -type f` (在`/home/user`下查找所有.py文件)。

### `which`
查找一个命令的可执行文件的完整路径。
- **示例**: `which python3`

### `whereis`
查找命令的二进制文件、源代码和手册页位置。

### `locate`
快速查找文件（基于一个预先生成的数据库）。比`find`快，但可能不是最新的。需`sudo updatedb`更新数据库。

---

## 4. 权限管理

### `sudo` (Superuser Do)
以超级用户(root)权限执行命令。

- +x 的意思是 "添加 (add) 执行 (execute) 权限"。

- **示例**: `sudo apt install nginx`

### `chmod` (Change Mode)
修改文件或目录的访问权限。
- **模式**:
    - **符号模式**: `u` (user), `g` (group), `o` (other), `a` (all); `+` (添加), `-` (移除), `=` (设置); `r` (读), `w` (写), `x` (执行)。
    - **数字模式**: `4` (读), `2` (写), `1` (执行)。三位数字分别代表u, g, o的权限总和。
- **-R**: 递归修改。
- **示例 (符号)**: `chmod +x my_script.sh` (为脚本添加可执行权限)。
- **示例 (数字)**: `chmod 755 my_script.sh` (设置为 `rwxr-xr-x` 权限)。

### `chown` (Change Owner)
修改文件或目录的所有者和所属用户组。
- **-R**: 递归修改。
- **示例**: `sudo chown -R www-data:www-data /var/www/my_blog`

---

## 5. 系统信息与进程管理

### `date`

查询当前时间

输出：

```
Thu  9 Oct 16:34:31 CST 2025
```

UTC 是协调世界时。如果在中国，看到的是 CST (China Standard Time)

### `timedatectl `

现代 Linux 系统使用 timedatectl 命令来管理时间和日期

#### `timedatectl status`

输出:

```bash
Local time: Thu 2025-10-09 16:35:26 CST
Universal time: Thu 2025-10-09 08:35:26 UTC
RTC time: n/a
Time zone: Asia/Shanghai (CST, +0800)
System clock synchronized: yes
NTP service: active
RTC in local TZ: no
```

- **Time zone**: 显示当前系统设置的时区。如果这里不是你在的时区，你的定时任务cron执行时间就会不符合你的预期！
- **System clock synchronized: yes**:  yes 表示树莓派已经成功通过网络时间协议（NTP）与世界标准时间服务器同步了，时间是绝对准确的。如果是 no，说明同步有问题。
- **NTP service: active**: 确认时间同步服务正在运行。

#### `timedatectl list-timezones`

列出所有可用的时区

使用 `grep` 过滤

```bash
timedatectl list-timezones | grep Asia/Shanghai
```

#### `sudo timedatectl set-timezone Asia/Shanghai`

设置时区为Asia/Shanghai

### `uname -a`
显示详细的系统内核信息。

### `df -h` (Disk Free)
以人类可读格式显示磁盘分区的使用情况。

### `du -h` (Disk Usage)
以人类可读格式显示目录或文件的磁盘占用大小。
- **-s**: 只显示总计大小。
- **示例**: `du -sh /var/log`

### `free -h`
以人类可读格式显示内存和交换空间的使用情况。

### `top` / `htop`
动态实时监控系统进程和资源占用。`htop`是彩色增强版，更直观（可能需`sudo apt install htop`）。

### `ps` (Process Status)
显示当前进程的快照。
- **`ps aux`**: 显示所有用户的全部进程的详细信息。
- **`ps -ef`**: 另一种常用格式。
- `ps aux | grep "pnpm run build"`

### `kill`
终止进程。
- **`kill <PID>`**: 发送终止信号 (SIGTERM)，允许进程优雅退出。
- **`kill -9 <PID>`**: 发送强制杀死信号 (SIGKILL)，进程立即被终止。
- **`killall <进程名>`**: 杀死所有同名进程。

### `pgrep`
根据名称或其他属性查找进程ID。
- **-f**: 匹配完整命令行参数。
- **示例**: `pgrep -f webui.py`

### `nvidia-smi`
**(GPU服务器必备)** 监控NVIDIA显卡状态。
- **`watch -n 1 nvidia-smi`**: 每秒刷新一次，实现动态监控。

### `netstat `

```
netstat -tulpn | grep :8080
```

查看对应端口进程

### `systemctll`

```
sudo systemctl status 
```

查看系统的服务状态

```
sudo systemctl status cron
```

查看定时器相关的系统运行服务

---

## 6. 网络操作

### `ip addr`
显示和管理网络接口地址（现代推荐）。

### `ifconfig`
显示和管理网络接口地址（旧版工具，可能需安装`net-tools`）。

### `ping`
测试与目标主机的网络连通性。

### `netstat`
显示网络连接、路由表、接口统计等信息（旧版工具）。
- **-tuln**: 查看TCP/UDP的监听端口。

### `ss`
`netstat`的现代替代品，速度更快。

- **`ss -tuln`**: 同上。

### `wget`
从URL下载文件。
- **-c**: 断点续传。
- **-O <文件名>`**: 指定保存的文件名。

### `curl`
功能强大的URL传输工具，可用于下载、API测试等。
- **示例**: `curl -L https://example.com/file`

  `-I`   选项表示只获取网页的头信息，速度更快

### `ssh`
安全地远程登录另一台主机。
- **示例**: `ssh myuser@192.168.1.100`

### `scp`
通过 SSH (Secure Shell) 协议在不同计算机之间安全地复制文件或目录。

- **-r**: 递归复制整个目录。
- **-P** (大写): 指定远程主机的 SSH 端口号（如果不是默认的22）。
- **-v**: 显示详细的连接和传输过程（Verbose模式），用于调试。
- **-C**: 启用压缩，可以在慢速网络上加快传输速度。

- **示例 (上传)**: `scp local_file.zip user@remote_ip:/remote/path/`

- **示例 (下载)**: `scp user@remote_ip:/remote/path/file.zip .`

- ```
  上传文件
  语法: scp [本地文件路径] [用户名]@[远程主机地址]:[远程目标路径]
  scp my_script.py alen@alen.local:~/scripts/
  上传目录
  语法: scp -r [本地目录路径] [用户名]@[远程主机地址]:[远程目标路径]
  scp -r ./my_blog_project/ alen@alen.local:~/
  ```

- **IPv6需要使用""**

  ```
  scp dist.zip "alen@[2001:da8:215:3c02:f868:7743:2c94:89fd]:~/blog/lingLong/lingLong/"
  ```

---

## 7. 软件包管理 (Debian/Ubuntu/Raspberry Pi OS)

### `apt`
高级软件包工具。
- **`sudo apt update`**: 更新本地软件包列表。
- **`sudo apt upgrade`**: 升级所有已安装的软件包。
- **`sudo apt full-upgrade`**: 升级软件包，并处理依赖关系变化（可能删除旧包）。
- **`sudo apt install <包名>`**: 安装软件包。
- **`sudo apt remove <包名>`**: 卸载软件包（保留配置文件）。
- **`sudo apt purge <包名>`**: 彻底卸载软件包（包括配置文件）。
- **`sudo apt autoremove`**: 移除不再需要的依赖包。
- **`apt search <关键词>`**: 搜索可用软件包。
- **`apt show <包名>`**: 显示软件包详细信息。

### `wget`

非交互式网络下载器。

- `wget <URL>`: 下载文件到当前目录。
- `wget -O <新文件名> <URL>`: 下载文件并指定新的文件名。
- `wget -P <目录> <URL>`: 下载文件到指定目录。
- `wget -c <URL>`: 断点续传，继续未完成的下载。
- `wget -b <URL>`: 后台下载文件，这对于下载大文件非常有用。
- `wget -q <URL>`: 静默模式，不输出任何下载过程信息。
- `wget --limit-rate=<速率> <URL>`: 限制下载速度（例如 100k, 1m）。
- `wget --no-check-certificate <URL>`: 忽略SSL证书验证（请谨慎使用）。
- **`wget -r -np -k <URL>`**: 递归下载整个网站（镜像）。-r 递归, -np 不追溯到父目录, -k 转换链接为本地链接。

---

## 8. 压缩与解压

### `tar` (Tape Archive)
打包与解包文件，通常与其他压缩工具结合使用。
- **-c**: 创建归档。
- **-x**: 提取归档。
- **-z**: 通过gzip进行压缩/解压 (.tar.gz)。
- **-j**: 通过bzip2进行压缩/解压 (.tar.bz2)。
- **-v**: 显示详细过程。
- **-f**: 指定归档文件名。
- **示例 (压缩)**: `tar -czvf archive.tar.gz /path/to/directory`
- **示例 (解压)**: `tar -xzvf archive.tar.gz`

### `zip` / `unzip`
处理`.zip`格式的压缩文件。
- **`zip -r archive.zip /path/to/directory`**
- **`unzip archive.zip`**

### `gzip` / `gunzip`
处理`.gz`格式的压缩文件（一次只能压缩单个文件）。

---

## 9. 会话管理与后台运行 (服务器必备)

### `tmux`
强大的终端复用器，用于创建持久化会话

使用此指令可以随时从当前会话中退出并关闭 SSH 窗口，而会话和在其中运行的所有程序（包括部署脚本）都会在服务器后台继续运行

- tmux new -s < 会话名 >:     新建并进入会话
- tmux ls:     列出所有会话
- tmux attach -t <会话名> (或 tmux a -t ...):      重新连接到会话
- tmux kill-session -t <会话名>：  强行停止部署
- **在会话内**:
    - `Ctrl+B` 然后 `D`:     脱离当前会话（程序在后台继续运行）
    - `Ctrl+B` 然后 `%`:     垂直分割窗口
    - `Ctrl+B` 然后 `"`:     水平分割窗口
    - `Ctrl+B` 然后 `方向键`:     在分割的窗格间移动

### `&` (Ampersand)
将一个命令放到后台运行。
- **`nohup <命令> &`**: (No Hang Up) 让命令在后台运行，并忽略挂起信号（即关闭终端后程序继续运行），输出重定向到`nohup.out`文件。
- **示例**: `nohup python my_long_running_task.py &`

### `jobs`
查看当前终端会话的后台任务。

### `fg` / `bg`
将后台任务切换到前台 (`fg`) 或让暂停的任务在后台继续运行 (`bg`)。

---

## 10. 帮助命令

### `man <命令>` (Manual)
显示命令的详细手册页。按 `q` 退出。
- **示例**: `man ls`

### `<命令> --help`
通常会显示命令的简明用法和选项列表。
- **示例**: `tar --help`

##  11.退出指令

### `exit`

可退出 ssh 或者 root模式

## 12.配置clash

1. ```bash
   nano ~/.bashrc
   ```

   进入配置

2. 添加

   ```bash
   export http_proxy=http://127.0.0.1:7890
   export https_proxy=http://127.0.0.1:7890
   export no_proxy="localhost,127.0.0.1,*.local,*.bupt.edu.cn"
   ```

   检查服务状态

   ```bash
   sudo systemctl status clash
   ```

   - /etc/systemd/system/clash.service: 这是服务的定义文件。
   - enabled: 这明确地表示，该服务已经被设置为**开机自启**。

4. 代替指令

   1. sudo systemctl start clash 代替 **clashctl on**
   2. sudo systemctl stop clash 代替 **clashctl off**
   3. 重启服务: sudo systemctl restart clash
   4. 设置开机自启: sudo systemctl enable clash
   5. 取消开机自启: sudo systemctl disable clash
   6. 查看状态: sudo systemctl status clash

## 13.登录docker

1. 配置docker主文件---**添加镜像源，不添加校园网别想上去了**

   ```
   sudo nano /etc/docker/daemon.json
   ```

2. 添加内容：镜像源

   ```json
   {
     "registry-mirrors": [
     "https://kbnxxtib546zgm.xuanyuan.run",
     "https://kbnxxtib546zgm.xuanyuan.dev",
     "https://docker.xuanyuan.me",
     "https://docker.1panel.live",
     "https://docker.m.daocloud.io",
     "https://docker-0.unsee.tech"
   ],
     "insecure-registries": [
     "https://kbnxxtib546zgm.xuanyuan.run",
     "https://kbnxxtib546zgm.xuanyuan.dev",
     "docker.xuanyuan.me"
   ],
     "dns": ["119.29.29.29", "114.114.114.114","8.8.8.8","223.223.223.223"]
   }
   ```

3. 安装`docker-compose`

   ```shell
   sudo apt-get install -y docker-compose-plugin
   ```

4. 配置代理配置文件:---全局代理

5. ```bash
   sudo nano /etc/systemd/system/docker.service.d/http-proxy.conf
   [Service]
   Environment="HTTP_PROXY=http://127.0.0.1:7890"
   Environment="HTTPS_PROXY=http://127.0.0.1:7890"
   ```

6. 重启systemd 配置

   ```bash
   sudo systemctl daemon-reload
   ```

7. 重启docker

   ```bash
   sudo systemctl restart docker
   ```

8. 查看docker容器

   ```bash
   docker ps -a
   ```

9. 查看日志

   ```bash
   docker logs sim_backend --tail 100
   docker logs sim_nginx --tail 100
   docker logs sim_backend -f
   docker logs sim_nginx -f
   ```

10. https://gh-proxy.com ----国内github镜像源

11. 在树莓派上配置 pnpm 镜像

    ```bash
    pnpm config set registry https://registry.npmmirror.com
    ```

12. 查看配置

    ```bash
    pnpm config get registry
    ```

13. 查看堆栈使用情况

     ```bash
     top -bn1 | grep node
     ```

14. cloudflared 配置证书

     ```
     /home/alen/.cloudflared/cert.pem
     ```

15. 查看容器是否构建文章

     ```bash
     docker exec sim_backend ls /code/lingLong/dist/posts/ | grep 测试
     ```

16. 清除无用的、悬空的旧镜像

    ```bash
    docker system prune
    ```

17. 查看已有的镜像文件

    ```bash
    docker images 
    ```

18. 清除指定镜像

    ```bash
    docker rm img_name 
    ```

19. 检查状态

    ```bash
    sudo systemctl status docker
    ```

    

    ---

    

## 14.配置cloudflare

1. 准备工作

     - 将域名DNS迁移到 Cloudflare（在阿里云修改NS记录）

     - 注册 Cloudflare 账号


  2. 在树莓派上安装 cloudflared

  3. 下载 ARM64 版本

     ```shell
     wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm64
     ```

  4. 安装

     ```shell
     sudo mv cloudflared-linux-arm64 /usr/local/bin/cloudflared
     sudo chmod +x /usr/local/bin/cloudflared
     ```

  5. 验证安装

      ```shell
      cloudflared --version
      ```

  6. 登录并创建隧道

       1. 登录 Cloudflare    

          ```shell
          cloudflared tunnel login
          ```

       2. 创建隧道（假设域名是 example.com）

          ```shell
          cloudflared tunnel create myblog
          ```

  7. 配置隧道路由

     ```shell
     cloudflared tunnel route dns myblog blog.example.com
     ```

  8. 创建配置文件

     ```shell
     sudo mkdir -p /etc/cloudflared
     sudo nano /etc/cloudflared/config.yml
     ```

       填入以下内容：

     ```shell
       tunnel: 你的tunnel ID
       credentials-file: /home/alen/.cloudflared/dawdawddwa12312311.json
     
       ingress:
         - hostname: 域名
           service: http://localhost:8090
         - service: http_status:404
     
     ```

  9. 设置开机自启

     ```
     sudo cloudflared service install
     sudo systemctl enable cloudflared
     sudo systemctl start cloudflared
     ```


## 15.定时器cron

### `crontab -l`

查询定时器任务列表

### `sudo systemctl status cron`

查询定时器cron状态

### `journalctl -u cron.service`

查询 journald 日志数据库的命令。查看 cron 服务的执行记录

- -u 是 --unit 的缩写，意思是“只显示这个特定服务单元的日志”。

```
journalctl | grep CRON
```

