# windows环境配置

因为已经有前面的环境了，所以试图在windows环境下配置，

首先重写了一下`makefile`文件，因为windows下的`make`命令和linux下的不一样，所以需要重新写一下。
这一般实在十分痛苦，让人不理解为什么微软当初设计的时候不和linux或者说unix统一一下。cmd和powershell之间也不甚兼容。

而且chatgpt似乎也不甚擅长这方面的工作，编写出的代码十分不能使用。
```makefile
.PHONY: lib, pybind, clean, format, all

all: lib

lib:
	@if not exist build mkdir build
	@cd build && cmake ..
	@cd build && $(MAKE)

format:
	python -m black .
	clang-format -i src/*.cc src/*.cu

clean:
	@if exist build rmdir /s /q build
	@del /q python/needle/backend_ndarray/ndarray_backend*.so

```
装完cmake后配上path，但是我才发现windows下的path居然需要重启电脑生效，或许有一些方法能使其生效，但是考虑到不同终端之间的交互，不如直接重启电脑。

后面出现的问题是发现我的Mingw32不兼容posix，搜索了一些资料，重新安装了mingw64，和vc++,但是还是这个问题，实在是受不了了，选择放弃windows环境，转向linux环境。


# linux环境配置
如果能重来，我一定选择在一开始就把我所有的编程环境都放在linux下，因为linux下的环境配置实在是太正常了，

令人伤心的是，刚刚才把我的wsl删除了，重新下载会来之后出现了很多奇怪的问题，而且网上的教程好多有问题

点名批评dns，![https://blog.csdn.net/Sys_tem123/article/details/128262394](https://blog.csdn.net/Sys_tem123/article/details/128262394)
这里面教你删文件找东西，其实是完全不需要的，wsl中的主机写死在了hosts中是127.0.1.1，我不知道他在那里搞什么东西

linux中的环境配置还是比较友善的，虽然中间一些奇怪的gpg什么的东西卡了我有一会


# 
不知道为什么，报错显示说我的needle包中出现的重复依赖，学了一遍python的packet管理之后发现没问题啊，也不知道为什么。
又去看了看pyb11这个东西，自己手懂编译出的文件能用到是，我要去看看为什么他的奇怪cmake编译出的东西不能用，我也真是伤心，卡了我半天。
执行下面这个奇怪的代码，话说为什么会有人把c++的源文件命名为.cc啊
```bash
c++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) ndarray_backend_cpu.cc -o ndarray_backend_cpu$(python3-config --extension-suffix)
```

后面找到问题了，他的cmake中每次找到的我的python环境都是我的base环境，我真是受不了了，明明pybind11都能找到我的conda环境，但是python环境还就那个倔强的一批，不知道为什么。他这两个东西的路径不一样也不做检查。
最后给他指定路径
```cmake
set(Python_ROOT_DIR /home/alhoa/anaconda3/envs/d2l/python3.9)
set(Python_EXECUTABLE /home/alhoa/anaconda3/envs/d2l/bin/python3.9)
set(Python_INCLUDE_DIR /home/alhoa/anaconda3/envs/d2l/include/python3.9)
set(Python_LIBRARY /home/alhoa/anaconda3/envs/d2l/lib/libpython3.9.so)
```

# 终于开始了


做了半天发现不能直接改那个东西，还需要重新调用一下make函数，

奇怪的是这里的broadcast 居然不copy也不用复制，直接改stride假装cast了但是问题是这样改的时候真的可以吗
还是说是因为什么奇怪的cow机制，但是我感觉好像没有啊，

get_item的计算还挺麻烦，需要思考一下如何计算对应的几个值。

compact 要求把一个array中的所有的值都放到一个array中，维护一个当前的数量和现在已经遍历的位置，每次计算位置并放入，同时根据stride来更新位置。

弄到一半发现有cuda的，考虑到重复了，仅完成矩阵乘法，写了半天，自信满满的写完了，跟答案的对了对感到十分伤心，好多超参数我都是乱取的，但是研究了一下发现还有好多访存方面的机制能提高速度，看来这头要学习的还不少。