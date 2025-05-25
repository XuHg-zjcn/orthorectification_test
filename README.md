# 正射校正测试
## 介绍
USGS EarthExplorer下载下来的解密影像没有坐标信息，更没有正射校正，只是底片复制品的扫描图像  
我在绘制OpenHistoricalMap，我觉得这些解密影像是很好的资料，处于公共领域，没有版权问题   
但是用QGIS的Georeferencer工具进行配准，不会考虑地形，不同海拔高度会造成漂移，会影响精度  
例如JOSM编辑器的Bing影像，都已经进过正射校正过了，与GPS轨迹相比山区误差也很小  
我想我应该也要把USGS解密影像进行一下正射校正，再当作编辑OpenHistorialMap的底图。  
我发现gdalwarp可以输入原始影像、RPC(有理多项式系数)和DEM数据，就能输出正射校正后的影像了，DEM数据可以使用SRTM，只要制作一个RPC文件就能完成校正了  
物理模型建模有点麻烦，我尝试直接使用多项式拟合，不断尝试降低参数个数并且使误差增大不明显，最终找到一个比较合适的多项式  

卫星上有两个相机，我想可以用来生成DEM数据，但还没有测试过  

这篇文章不是当时记录下来的，隔了几个星期突然想到还有一些代码没有上传，还要写点文档介绍一下，所以是按照回忆写下  


## 图像对齐
`match_test.py`是一个SIFT+RANSAC算法的立体图像对齐程序。
我用`./match_test.py -a D3C1208-200141A020_h.tif -b D3C1208-200141F020_c.tif -m match_x.jpg -3 match3d.jpg`进行测试，除了山区附近的有误差，用3D眼镜查看效果很好，能看到建筑物和树木的凸起。

也能匹配一些SPOT4和SPOT5影像，时间间隔1年左右的两幅影像能匹配，时间间隔太长就不行。

- `D3C1208-200141A020_h.tif`和`D3C1208-200141F020_c.tif`分别是从[USGS EarthExplorer](https://earthexplorer.usgs.gov/)上下载`D3C1208-200141A020.tgz`和`D3C1208-200141F020.tgz`压缩包解压后得到
- SPOT影像是从[CNES Spot World Heritage](https://regards.cnes.fr/user/swh/modules/60)下载的

## 版权和许可证
Copyright (C) 2024 Xu Ruijun  
代码使用GNU通用公共许可证第三版或更新授权  
  
本README.md文件使用以下许可证  
```
Copyright (C) 2024 Xu Ruijun

Copying and distribution of this README file, with or without modification,
are permitted in any medium without royalty provided the copyright
notice and this notice are preserved.  This file is offered as-is,
without any warranty.
```
