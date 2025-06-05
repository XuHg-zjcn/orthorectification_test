# 正射校正测试
## 背景
USGS EarthExplorer下载下来的解密影像的没有进行坐标变换（但在元数据里有大致的四角坐标），更没有正射校正，只是底片复制品的扫描图像  
我在绘制[OpenHistoricalMap](https://www.openhistoricalmap.org/)的绍兴区域，这些影像是很好的资料，[处于公共领域](https://www.usgs.gov/information-policies-and-instructions/copyrights-and-credits)，不会造成侵权  
但是用QGIS的[Georeferencer](https://docs.qgis.org/3.40/en/docs/user_manual/managing_data_source/georeferencer.html)工具进行配准，不会考虑地形，不同海拔高度会造成漂移，会影响精度  
例如JOSM编辑器的Bing影像，都已经进过正射校正过了，与GPS轨迹相比山区误差也很小  
我想我应该也要把USGS解密影像进行一下正射校正，再当作编辑OpenHistorialMap的底图。  
我发现gdalwarp可以输入原始影像、RPC(有理多项式系数)和DEM数据，就能输出正射校正后的影像了，DEM数据可以使用SRTM，只要制作一个RPC文件就能完成校正了  
物理模型建模有点麻烦，我尝试直接使用多项式拟合，不断尝试降低参数个数并且使误差增大不明显，最终找到一个比较合适的多项式  

卫星上有两个相机，我想可以用来生成DEM数据，但还没有测试过  

这篇文章不是当时记录下来的，隔了几个星期突然想到还有一些代码没有上传，还要写点文档介绍一下，所以是按照回忆写下

人工配准太麻烦了，现在尝试用OpenCV进行图像与图像的匹配。
以后只用人工配准一次（或使用其他已配准的影像）就能自动将其他重叠区域的影像配准了

## 导入到数据库
命令`./import_img.py <files> <files>...`会将图像的元数据记录到数据库（位于`data/imagery.db`），用于后续处理

## 自动匹配影像
命令`./match_levels.py <file>`将低分辨率影像与高分辨率影像匹配，命令行参数是低分辨率影像的路径，会从数据库按元数据中的覆盖范围的重叠高分辨率影像进行匹配

## 手动匹配影像
`match_imgpair.py`是一个SIFT+RANSAC算法的立体图像对齐程序，支持使用裁剪黑边、降低分辨率、拉普拉斯变换、膨胀进行预处理。
一些命令行选项可重复多次出现，与位置相关，决定每个图像的处理方式，如`--laplace`,`--dilsize`,`--maxpixel_sift`,`--cutblack_topbottom`,`--cutblack_leftright`（如果出现在指定任何图像之前(`-a`,`-b`选项之前)，将会代替该选项的默认值，如果这些选项出现在指定图像文件后，只会更改该图像文件的选项。
### 测试实例
- `./match_imgpair.py  --maxpixel_sift 1e7 --cutblack_topbottom -a data/D3C1208-200141A020_h.tif -b data/D3C1208-200141F020_c.tif -m data/match_x.jpg -3 match3d.jpg`能匹配2382个点，除了山区附近的有误差，用3D眼镜查看效果很好，能看到建筑物和树木的凸起。
- `./match_imgpair.py --laplace --dilsize 8 --maxpixel_sift 1e6 --threshold_m1m2_ratio 0.9 --cutblack_topbottom -a data/D3C1208-200141A020_h.tif -b data/D3C1215-401419A011_e.tif -m data/match_1974_1979.jpg`能匹配260点

也能匹配一些SPOT4和SPOT5影像，时间间隔超过约1年的影像需要使用拉普拉斯变换和膨胀预处理才能成功匹配
- `./match_imgpair.py -a data/003-008_S5_295-290-0_2011-11-24-02-41-14_HRG-2_A_DT_JK/SCENE01/IMAGERY.TIF -b data/003-004_S5_295-290-0_2012-01-04-02-52-09_HRG-2_A_DT_NO/SCENE01/IMAGERY.TIF -m data/match_spot_2011_2012.jpg`能匹配770个点
- `./match_imgpair.py  --laplace --dilsize 8 --maxpixel_sift=1e6 --threshold_m1m2_ratio 0.87 -a data/003-008_S5_295-290-0_2011-11-24-02-41-14_HRG-2_A_DT_JK/SCENE01/IMAGERY.TIF -b data/004-010_S5_295-290-0_2014-02-01-02-25-09_HRG-2_B_DT_NO/SCENE01/IMAGERY.TIF -m data/match_spot_2011_2014.jpg`能匹配30个点

## 大致处理流程
1. 根据元数据找出相交的影像
2. 初步匹配影像，找出投影矩阵
3. 进一步匹配影像，考虑视差分布
4. 找出对应的GCP点
5. 生成RPC系数文件
6. 运行`gdalwarp`

## 数据来源
- `D3C1208-200141A020_h.tif`和`D3C1208-200141F020_c.tif`等影像是从[USGS EarthExplorer](https://earthexplorer.usgs.gov/)上下载Declass影像压缩包解压后得到
- SPOT影像是从[CNES Spot World Heritage](https://regards.cnes.fr/user/swh/modules/60)下载的

这些数据都在外国网站，如果下载较慢或无法下载，也可以联系我来获取，联系方式见[我的Codeberg Page](https://techhorse.codeberg.page/)

## 参考资料
- 知乎：[python遥感影像配准以及生成配准后影像的RPC系数文件(附源码与打包好的软件)](https://zhuanlan.zhihu.com/p/24076276139)
- 知乎：[【转载】机器视觉：SIFT Matching with RANSAC](https://zhuanlan.zhihu.com/p/588670464)
- [Automated Processing of Declassified KH-9 Hexagon Satellite Images for Global Elevation Change Analysis Since the 1970s](https://www.frontiersin.org/journals/earth-science/articles/10.3389/feart.2020.566802/full)
- 上一篇文章提到了NASA的[Ames Stereo Pipeline](https://github.com/NeoGeographyToolkit/StereoPipeline)软件，我看了文档有[关于处理KH-9影像的内容](https://stereopipeline.readthedocs.io/en/latest/examples/historical.html#declassified-satellite-images-kh-9)，我还没有尝试过此款软件
- Orfeo Toolbox的[正射校准文档](https://www.orfeo-toolbox.org/CookBook/Applications/app_OrthoRectification.html)，目前只能对SPOT5影像进行正射校准

## 版权和许可证
Copyright (C) 2024-2025 Xu Ruijun  
代码使用GNU通用公共许可证第三版或更新授权  
  
本README.md文件使用以下许可证  
```
Copyright (C) 2024-2025 Xu Ruijun

Copying and distribution of this README file, with or without modification,
are permitted in any medium without royalty provided the copyright
notice and this notice are preserved.  This file is offered as-is,
without any warranty.
```
