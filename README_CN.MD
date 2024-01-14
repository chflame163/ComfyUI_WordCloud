# ComfyUI Word Cloud��
ComfyUI���ɴ���ͼ��Ĳ����
![image](image/comfy_wordcloud_advance.png)

## ���£�
* ����RGB Color Picker�ڵ㣬ѡ����ɫ���ӷ��㡣

* ͨ���༭�����Ŀ¼�µ�font_dir.ini���û������Զ�������Ŀ¼��ÿ������comfyUIʱ�����Ŀ¼��*.ttf��*.otf�ļ����ռ���ʾ�ڲ��font_pathѡ�����С�
font_dir.iniĬ����windowsϵͳ����Ŀ¼(C:\Windows\fonts)��
����Զ����Ŀ¼��Ч���������Դ�fontĿ¼�����Ŀ¼����һ��Alibaba-PuHuiTi-Heavy.ttf�ļ�����Ȩ��������Ͱͣ��й������޹�˾�������κθ��˺���ҵ���ʹ�á�



### ��Ҫ���ѣ�����֮ǰ����������ľɰ�ڵ㣬��Ҫ�����������塣
* �������ú�font_dir.ini������comfyUI�����ع���������WordCloud�ڵ��font_pathѡ����������ѡ�����壬���ɱ��ⱨ����

* ���θ��»���[ZHO-ZHO-ZHO](https://github.com/ZHO-ZHO-ZHO/ComfyUI-Text_Image-Composite)�Ľ���Ͱ�����


## �ڵ�˵����

### Word Cloud:
�����ı��������ɴ���ͼ�񣬴ʵĴ�С���Ƶ��أ�Ƶ��Խ������Խ�󡣿ɶ�����ɫ���趨�ص�ʣ��趨�ų��ʵȡ�֧�������alphaͨ����ͼ�����ɾ���������״�Ĵ���ͼ��

ѡ��˵����   
* color_ref_image: �����ͼ����Ϊ������ɫ�Ĳο���
* mask_image: �����ͼ����Ϊ������״�������������alphaͨ����ʹ��ͨ����Ϊ����������ͨ���İ���ɫ������������״����ɫ�Ĳ��ֽ����ų���
* text: �˴����ı����ݽ��ֽ�Ϊ������ʣ���Щ������Ϊ���Ƶ�Ԫ�ء����ʳ���Ƶ��Խ�ߣ��ڴ���������Խ��
* width: ����ͼ��Ŀ��ȡ�(�����mask_image���룬��ʹ��mask_image�ĳߴ磬��������ֵ������)
* height: ����ͼ��ĸ߶ȡ�(�����mask_image���룬��ʹ��mask_image�ĳߴ磬��������ֵ������)
* scale: �Ŵ�ϵ�����������ɵ�ͼ���С����width��height�趨��ֵ�������ˡ�
* margin: �հױߴ�С��
* font_path: �����ļ���
* min_font_size: ����Ԫ����ʾ����Сֵ��
* max_font_size������Ԫ����ʾ�����ֵ��
* relative_scaling: �����е���Ԫ�ص���Դ�С��ֵԽ����ɢ��Խ�ߡ�
* colormap: ������ɫ������ʹ�õ���Matplotlib �ṩ�Ķ���Ԥ����colormap��(�����color_ref_image���룬������ñ�����)
* background_color: ������ɫ��ʹ��16����RGB��ʽ������(���transparent_background�趨ΪTrue��������ñ�����)
* transparent_background: �����Ƿ�͸�����˴���ΪTrue�������alphaͨ����ͼ��
* prefer_horizontal: ����ˮƽ�����ʡ����0(ȫ����)�����1(ȫ����)��
* max_words: �������������
* repeat: ������û�дﵽ�������ʱ�Ƿ������ظ���
* include_numbers: ����Ԫ���Ƿ�������֡�
* random_state�����ƴ������ɹ���������������������ӡ���Ϊ-1Ϊÿ�����������ֵ��ÿ�ι̶���
* stopwords: ���δʡ��趨�ڴ˴��ĵ��ʲ�����ͼ�г��֡�ÿ����֮���ö���(��Ӣ�ľ���)��ո�ֿ���
* contour_width: �������ȡ���mask_image�������Ч��
* contour_color: ������ɫ��ʹ��16����RGB��ʽ��������mask_image�������Ч��
* keynote_words: �ص�ʡ������趨�ĵ��ʽ�����Ŵ���stopwords���趨��ͬ�����ʵĳ��⡣ÿ����֮���ö���(��Ӣ�ľ���)��ո�ֿ���
* keynote_weight: �ص�ʼ�Ȩ����ֵԽ���ص�����Խ��

�����
image(֧��alphaͨ��)

### RGB Color Picker
�޸���[mtb nodes](https://github.com/melMass/comfy_mtb)��Web Extensions����ɫ����ѡȡ��ɫ�������RGBֵ��

ѡ��˵����   
* mode�� �����ʽ����ʮ������(HEX)��ʮ����(DEC)��ѡ��  

����� 
string�ַ�����

### Load Text File��
��ָ��·�������ı��ļ����趨ʹ��utf-8���롣

ѡ��˵����   
* path�� txt�ļ�·����   

����� 
string�ַ�����

## ʹ��ʾ����

![image](image/comfy_wordcloud_simple.png)
��workflowĿ¼����json��ʽ�Ĺ�����ʾ���ļ���

## ��װ������  

* ��CompyUI���Ŀ¼(���硰CompyUI\custom_nodes\��)�д�cmd���ڣ�����```git clone https://github.com/chflame163/ComfyUI_WordCloud.git```��װ���������ؽ�ѹzip�ļ������õ����ļ��и��Ƶ� ComfyUI\custom_nodes\
* ��װ������������Դ������ComfyUI\custom_nodes\ComfyUI_WordCloud ���Ŀ¼λ�ô�cmd���ڣ������������  
```..\..\..\python_embeded\python.exe -m pip install -r requirements.txt```
* ���´�ComfyUI��