# ASAT-DL
A Method for Processing Static Analysis Alarms Based on Deep Learning.

首先需要部署好Defects4J数据集，然后下载静态分析工具SpotBugs，接着就能运行所有的Python脚本啦。部分脚本的说明如下：

`get_info`：获取Defects4J数据集中的bugs信息，生成log文件。

`checkout_version.py`：从Defects4J数据集中导出所有buggy版本的源代码到工作目录~/defects4j/tmp。

`compile_and_test.py`：编译工作目录中共835个子目录的源代码并进行测试。

`export_metadata.py`：生成了{pid}-metadata.csv，统计了一些警报信息。

`spotbugs_report.py`：用SpotBugs在835个target_classes中的源代码进行静态分析，生成152518条静态分析警报。

`get_sourcecode.py`: 根据警报信息提供的代码行，提取警报相应的源代码，具体的代码行数没有统计，上百万条。

`generate_csv.py`：根据html文件中的警报信息提取特征生成csv文件。

如果能按照官方库 [defects4j](https://github.com/rjust/defects4j) 配置好，那所有脚本就可以直接用啦。但是因为墙的原因，我的同胞可能会失败，简单来说就是：

```bash
$ git clone https://github.com/rjust/defects4j
$ cd defects4j
# cpanm --installdeps .
# 因为墙的原因 cpanm --installdeps . 会失败 建议这步用网易源
$ cpanm --mirror http://mirrors.163.com/cpan --mirror-only --installdeps .
# $ ./init.sh
# 因为墙的原因 ./init.sh 会特别特别慢 我这里弄了个manual_download 来避免init.sh中调用download_url
$ cd manual_download && ./init.sh
$ export PATH=$PATH:"path2defects4j"/framework/bin
```

> manual_download链接: https://pan.baidu.com/s/19ZswcAd8YRLiJQUXlj3jqw?pwd=7777 提取码: 7777 复制这段内容后打开百度网盘手机App，操作更方便哦 
> --来自百度网盘超级会员v8的分享

安装过程中如果遇到以下错误，可以采取相应措施解决：

- `x86_64-linux-gnu-gcc: not found`

  ```bash
  $ sudo apt-get update
  $ sudo apt-get install build-essential
  ```

- `defects4j info -p Lang`时报错， `Can't locate DBI.pm in @INC`，则需要安装`Perl`模块 `DBI.pm`才能继续使用`defects4j`。

  ```bash
  $ sudo apt-get install libdbi-perl
  ```

具体的实验结果在不同配置的机器上可能有细微差别！

按照步骤来不可能复现失败的，`.ipynb`文件就是我们在实验室服务器上运行代码得到的。
