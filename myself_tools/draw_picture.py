import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os
import sys
import pandas as pd

class Sub_pic_info:
    def __init__(self, sub_pic_name, x_label, y_label, x_values, y_values, line_name, show_max_min=None, x_int_map_str_dict={}):
        self.sub_pic_name = sub_pic_name
        self.x_label = x_label
        self.y_label = y_label
        self.x_values = x_values
        self.y_values = y_values
        #如果有多条线就需要给一个数组区分每一条线的名称，只有一条线则不用输入
        self.line_name = line_name
        #由于x轴只能显示数字，如果说想把数字隐射成字符串，可以给一个dict表示映射关系
        self.x_int_map_str_dict = x_int_map_str_dict
        self.show_max_min = show_max_min
        self.match_shape_value()

        #确保line_name是一维数组，x_values, y_values是二维数组，且每一条线都有名字，每一条线的x，y轴点的个数相同
    def match_shape_value(self):
        if isinstance(self.line_name, str):
            self.line_name = [self.line_name]
        if not isinstance(self.x_values[0], list):
            self.x_values = [self.x_values]
        if not isinstance(self.y_values[0], list):
            self.y_values = [self.y_values]
        if self.show_max_min == None:
            self.show_max_min = [2]*len(self.line_name)
        if not isinstance(self.show_max_min, list):
            self.show_max_min = [self.show_max_min]
        assert isinstance(self.x_int_map_str_dict, dict)
        assert len(self.x_values) == len(self.y_values) == len(self.line_name)
        for index, x_arr in enumerate(self.x_values):
            assert len(x_arr) == len(self.y_values[index])

#获取整体图的大小
def get_total_fig_size(subpic_rows, subpic_cols, sub_fig_high, sub_fig_weight, space_between_sub_pic):
    pic_high = subpic_rows * (1 + space_between_sub_pic) * sub_fig_high
    pic_weight = subpic_cols * (1 + space_between_sub_pic) * sub_fig_weight
    return pic_weight, pic_high

#获取x轴的cue
def get_x_label_cue(map_dict):
    map_dict = dict(sorted(map_dict.items()))
    x_label = '('
    for num_str, map_str in map_dict.items():
        x_label += (num_str + ':' + map_str + '|')
    x_label = x_label.rstrip('|') + ')'
    return x_label

def get_boundary_info(x_values, y_values):
    #所有线的显示范围,初始化为系统最大和最小值
    value_total_min_y = total_min_int_x = total_min_int_y = sys.maxsize
    value_total_max_y = total_max_int_x = total_max_int_y = -sys.maxsize - 1
    for line_index, line_y_arr in enumerate(y_values):
        line_x_arr = x_values[line_index]
        min_x = min(line_x_arr)
        max_x = max(line_x_arr)
        min_y = min(line_y_arr)
        max_y = max(line_y_arr)
        if min_x < total_min_int_x:
            total_min_int_x = min_x
        if max_x > total_max_int_x:
            total_max_int_x = max_x
        if int(min_y) < total_min_int_y:
            total_min_int_y = int(min_y)
        if int(max_y) + 1 > total_max_int_y:
            total_max_int_y = int(max_y) + 1
        if min_y < value_total_min_y:
            value_total_min_y = min_y
            record_min_y_xvalue = line_x_arr[line_y_arr.index(min_y)]
        if max_y > value_total_max_y:
            value_total_max_y = max_y
            record_max_y_xvalue = line_x_arr[line_y_arr.index(max_y)]
    return value_total_min_y, total_min_int_x, total_min_int_y, record_min_y_xvalue, value_total_max_y, total_max_int_x, total_max_int_y, record_max_y_xvalue

def get_line_cue(min_y, max_y, min_index, max_index, show_flag = 2):
    if show_flag == 2:
        return f"{min_y:.4f}({min_index})/{max_y:.4f}({max_index})"
    elif show_flag == 0:
        return f"{min_y:.4f}({min_index})"
    elif show_flag == 1:
        return f"{max_y:.4f}({max_index})"

def match_y_position(y_dot_arr):
    gap = 0.4
    or_gap = 0.5
    map_y_dot_dict = {}
    for float_num in y_dot_arr:
        if float_num - int(float_num) != 0:
            if float_num - int(float_num) < or_gap:
                map_y_dot_dict[str(round(float_num, 4))] = round(float_num + gap, 4)
            elif int(float_num) + 1 - float_num < or_gap:
                map_y_dot_dict[str(round(float_num, 4))] = round(float_num - gap, 4)
            else:
                map_y_dot_dict[str(round(float_num, 4))] = round(float_num, 4)
    return map_y_dot_dict

def draw_sub_pic(pics, save_path, pic_rows, pic_cols, pic_name = None, smooth = False):
    num_sub_pics = len(pics)
    assert num_sub_pics <= (pic_rows * pic_cols)
    space_between_sub_pic = 0.2
    pic_weight, pic_high = get_total_fig_size(pic_rows, pic_cols, 4, 6, space_between_sub_pic)
    fig, axes = plt.subplots(pic_rows, pic_cols, figsize=(pic_weight, pic_high), squeeze=False)
    if pic_name != None:
        fig.suptitle(pic_name)
    #如果两个都等于一就手动加到二维
    if isinstance(axes, list):
        axes = [axes]
    #调整垂直和水平间距,参数为占子图比例
    fig.subplots_adjust(hspace=space_between_sub_pic, wspace=space_between_sub_pic)
    for i, ax in enumerate(axes.flat):
        if i < num_sub_pics:
            pic = pics[i]
            assert isinstance(pic, Sub_pic_info)
            show_max_min_arr = pic.show_max_min
            ax.set_title(pic.sub_pic_name)
            
            #所有线的显示范围,初始化为系统最大和最小值
            value_total_min_y, total_min_int_x, total_min_int_y, record_min_y_xvalue, value_total_max_y, total_max_int_x, total_max_int_y, record_max_y_xvalue = \
            get_boundary_info(pic.x_values, pic.y_values)
            y_fontzise = 8
            for line_index, line_y_arr in enumerate(pic.y_values):
                line_x_arr = pic.x_values[line_index]
                min_y = min(line_y_arr)
                min_index = line_x_arr[[i for i, x in enumerate(line_y_arr) if x == min_y][0]]
                max_y = max(line_y_arr)
                max_index = line_x_arr[[i for i, x in enumerate(line_y_arr) if x == max_y][0]]
                #如果存在映射提示就在线名称后显示映射信息
                show_flag = show_max_min_arr[line_index]
                line_cue = get_line_cue(min_y, max_y, min_index, max_index, show_flag)
                if len(line_y_arr) > 1:
                    #线性图
                    if smooth:
                        df = pd.DataFrame({'x': line_x_arr, 'y': line_y_arr})
                        df['y_smooth'] = df['y'].rolling(window=5, min_periods=1).mean()
                        ax.plot(df['x'], df['y_smooth'], label = line_cue + pic.line_name[line_index])
                    else:
                        ax.plot(line_x_arr, line_y_arr, label = line_cue + pic.line_name[line_index])
                else:
                    #如果只有一个点就画点图
                    ax.scatter(line_x_arr[0], line_y_arr[0], marker='o', label = pic.line_name[line_index])
                #调整图例文字大小
                ax.legend(fontsize=y_fontzise)
            # #在最高点和最低点画一条平行于X轴的线
            # ax.hlines(value_total_min_y, total_min_int_x, record_min_y_xvalue, color='grey', linestyle='--', linewidth=1)
            # ax.hlines(value_total_max_y, total_min_int_x, record_max_y_xvalue, color='grey', linestyle='--', linewidth=1)
            # ax.plot([x_value, x_value], [y_start, y_end], color='grey', linewidth=1)
            # ax.plot([x_value, x_value], [y_start, y_end], color='grey', linewidth=1)
            x_label = pic.x_label
            if len(pic.x_int_map_str_dict) > 0:
                x_label += get_x_label_cue(pic.x_int_map_str_dict) 
                ax.vlines(total_max_int_x - len(pic.x_int_map_str_dict), total_min_int_y, total_max_int_y, color='red', linestyle=':', linewidth=1)
            #x,y轴的名称
            ax.set_xlabel(x_label)
            #纵坐标竖着写
            ax.set_ylabel(pic.y_label, rotation=90)

            # y轴显示小数刻度位置调整
            # map_y_dot_dict = match_y_position([value_total_min_y, value_total_max_y])

            # if len(map_y_dot_dict) != 0:
            #     for show_str, position in map_y_dot_dict.items():
            #         # 在指定位置添加文字
            # ax.text(min_index, min_y, round(min_y, 3), ha='center', va='center', fontsize=y_fontzise, color='black')
            # ax.text(max_index, max_y, round(max_y, 3), ha='center', va='center', fontsize=y_fontzise, color='black')

            ax.tick_params(axis='y', labelsize=y_fontzise)
            ax.tick_params(axis='x', labelsize=y_fontzise)

            # 设置横，纵坐标轴显示范围
            ax.set_xlim(total_min_int_x, total_max_int_x)
            ax.set_ylim(total_min_int_y, total_max_int_y)
            # 使用 MaxNLocator 自适应设置 Y 轴刻度
            ax.yaxis.set_major_locator(MaxNLocator(nbins=3, prune='both'))
            # 使用 MaxNLocator 自适应设置 X 轴刻度
            ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5, prune='both'))

            # 获取端点刻度
            xtick_now = list(ax.get_xticks())
            x_gap = xtick_now[-1] - xtick_now[0]
            x_ticks = xtick_now
            if (xtick_now[0] - total_min_int_x) / x_gap < 0.05:
                x_ticks = [total_min_int_x] + x_ticks[1:]
            else:
                x_ticks = [total_min_int_x] + x_ticks
            if (total_max_int_x - xtick_now[-1]) / x_gap < 0.05:
                x_ticks = x_ticks[:-1] + [total_max_int_x]
            else:
                x_ticks = x_ticks + [total_max_int_x]
            ytick_now = list(ax.get_yticks())
            y_gap = ytick_now[-1] - ytick_now[0]
            y_ticks = ytick_now
            if (ytick_now[0] - total_min_int_y) / y_gap < 0.08:
                y_ticks = [total_min_int_y] + y_ticks[1:]
            else:
                y_ticks = [total_min_int_y] + y_ticks
            if (total_max_int_y - ytick_now[-1]) / y_gap < 0.08:
                y_ticks = y_ticks[:-1] + [total_max_int_y]
            else:
                y_ticks = y_ticks + [total_max_int_y]

            # 去重并排序
            ax.set_xticks(sorted(set(x_ticks)))
            ax.set_yticks(sorted(set(y_ticks)))

            # 添加区分不同颜色的线的图例
            ax.legend()
        else:
            # 如果图不够填充整个subplot，删除多余的subplot
            fig.delaxes(ax)
    if not os.path.exists(os.path.dirname(save_path)):
        os.mkdir(os.path.dirname(save_path))
    #让图片自适应四周空白
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Save pic to {os.path.abspath(save_path)}")