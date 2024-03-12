f = open('D:\MyWork\Program\MathModeling\\2023_C\data\word.txt','r',encoding='utf-8')
sta = [0 for i in range(26)]
freq = [0 for i in range(26)]
for i in range(5500):
    buf = f.readline()
    for j in buf:
        if j == 'a' or j == 'A':
            sta[0] = sta[0] + 1
        elif j == 'b' or j == 'B':
            sta[1] = sta[1] + 1
        elif j == 'c' or j == 'C':
            sta[2] = sta[2] + 1
        elif j == 'd' or j == 'D':
            sta[3] = sta[3] + 1
        elif j == 'e' or j == 'E':
            sta[4] = sta[4] + 1
        elif j == 'f' or j == 'F':
            sta[5] = sta[5] + 1
        elif j == 'g' or j == 'G':
            sta[6] = sta[6] + 1
        elif j == 'h' or j == 'H':
            sta[7] = sta[7] + 1
        elif j == 'i' or j == 'I':
            sta[8] = sta[8] + 1
        elif j == 'j' or j == 'J':
            sta[9] = sta[9] + 1
        elif j == 'k' or j == 'K':
            sta[10] = sta[10] + 1
        elif j == 'l' or j == 'L':
            sta[11] = sta[11] + 1
        elif j == 'm' or j == 'M':
            sta[12] = sta[12] + 1
        elif j == 'n' or j == 'N':
            sta[13] = sta[13] + 1
        elif j == 'o' or j == 'O':
            sta[14] = sta[14] + 1
        elif j == 'p' or j == 'P':
            sta[15] = sta[15] + 1
        elif j == 'q' or j == 'Q':
            sta[16] = sta[16] + 1
        elif j == 'r' or j == 'R':
            sta[17] = sta[17] + 1
        elif j == 's' or j == 'S':
            sta[18] = sta[18] + 1
        elif j == 't' or j == 'T':
            sta[19] = sta[19] + 1
        elif j == 'u' or j == 'U':
            sta[20] = sta[20] + 1
        elif j == 'v' or j == 'V':
            sta[21] = sta[21] + 1
        elif j == 'w' or j == 'W':
            sta[22] = sta[22] + 1
        elif j == 'x' or j == 'X':
            sta[23] = sta[23] + 1
        elif j == 'y' or j == 'Y':
            sta[24] = sta[24] + 1
        elif j == 'z' or j == 'Z':
            sta[25] = sta[25] + 1

        if j == '[':
            break
        pass

print('5498个词汇中，各字母出现的次数分别为：\n')
asc = 97
for i in range(26):
    if i < 25:
        print("%c" % asc,':',sta[i],end='   ')
        if (i + 1) % 5 == 0:
            print('\n')
    else:
        print("%c" % asc,':',sta[i])
    asc = asc + 1

print('\n')

sum = 0
for i in sta:
    sum = sum + i

for i in range(26):
    freq[i] = round(sta[i] / sum,4)
asc = 97
print('各字母出现的频率分别为：\n')
for i in range(26):
    if i < 25:
        print("%c" % asc,':',freq[i],end='   ')
        if (i + 1) % 5 == 0:
            print('\n')
    else:
        print("%c" % asc,':',freq[i])
    asc = asc + 1

f.close()
import pyecharts.options as opts
from pyecharts.charts import Line
from pyecharts.faker import Faker
import string

# 生成字母序列 a-zA-Z
alphabet = list(string.ascii_lowercase)

c = (
    Line()
    .add_xaxis(xaxis_data=alphabet)
    .add_yaxis(
        "Frequency",
        freq,
        symbol="triangle",
        symbol_size=20,
        linestyle_opts=opts.LineStyleOpts(color="green", width=4, type_="dashed"),
        itemstyle_opts=opts.ItemStyleOpts(
            border_width=3, border_color="yellow", color="blue"
        ),
    )
    .set_global_opts(title_opts=opts.TitleOpts(title="Line-ItemStyle"))
    .render("line_itemstyle.html")
)
