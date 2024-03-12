import pyecharts.options as opts
from pyecharts.charts import Line
from pyecharts.faker import Faker
import string

client = ['UNODC','INTERPOL','IMO','UNEP','ICCWC','EP','Customs','WLE','CBP','WWF','IUCN','WildAid','Traffic','CI','AsstrA','WILDLABS','NAT GEO']
score = [0.647,0.852,0.401,0.595,0.873,0.499,0.602,0.808,0.594,0.648,0.615,0.365,0.401,0.544,0.031,0.134,0.554]

c = (
    Line(init_opts=opts.InitOpts(width="800px", height="400px"))
    .add_xaxis(xaxis_data=client)
    .add_yaxis(
        "Score",
        score,
        symbol="circle",
        symbol_size=15,
        linestyle_opts=opts.LineStyleOpts(color="green", width=3, type_="dashed"),
        itemstyle_opts=opts.ItemStyleOpts(
            border_width=2, border_color="yellow", color="blue"
        ),
    )
    .set_series_opts(
        label_opts=opts.LabelOpts(is_show=True, position="top",border_width=4),
    )
    .set_global_opts(
        # title_opts=opts.TitleOpts(title="Line-ItemStyle"),
        xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=45)),  # 设置 X 轴标签为竖直方向
        )

    .render("client-score.html")
)