如果能按照官方库 [defects4j](https://github.com/rjust/defects4j) 配置好，那直接用就可以啦。但是因为墙的原因，我的同胞可能会失败，简单来说就是

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
> --来自百度网盘超级会员v7的分享
>

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
