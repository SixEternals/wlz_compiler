from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
from xml.sax.saxutils import escape


OUT = Path(__file__).resolve().parents[1] / "doc" / "项目介绍PPT.pptx"
EMU = 914400
W, H = 13.333, 7.5
NS = {
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}


def tag(prefix, name, attrs="", body=""):
    return f"<{prefix}:{name}{(' ' + attrs) if attrs else ''}>{body}</{prefix}:{name}>"


def x(v):
    return str(int(v * EMU))


def color(value):
    return f'<a:srgbClr val="{value}"/>'


def run(text, size=16, fill="D9E4EF", bold=False, font="Aptos"):
    attrs = f'lang="zh-CN" sz="{size * 100}"' + (' b="1"' if bold else "")
    props = f'<a:rPr {attrs}><a:solidFill>{color(fill)}</a:solidFill><a:latin typeface="{font}"/><a:ea typeface="Microsoft YaHei"/></a:rPr>'
    return f'<a:r>{props}<a:t>{escape(text)}</a:t></a:r>'


def paragraph(text, size=16, fill="D9E4EF", bold=False, align="l", bullet=False):
    ppr = f'<a:pPr algn="{align}">' + ('<a:buChar char="•"/>' if bullet else '') + '</a:pPr>'
    return f'<a:p>{ppr}{run(text, size, fill, bold)}<a:endParaRPr lang="zh-CN" sz="{size * 100}"/></a:p>'


def text_box(x0, y0, w0, h0, text, size=16, fill="D9E4EF", bold=False, align="l", valign="top", margin=0.08):
    paras = text if isinstance(text, str) else "".join(text)
    anchor = {"top": "t", "middle": "ctr", "bottom": "b"}.get(valign, valign)
    body = (f'<a:bodyPr lIns="{x(margin)}" rIns="{x(margin)}" tIns="{x(margin)}" bIns="{x(margin)}" '
            f'vert="horz" anchor="{anchor}" wrap="square"/><a:lstStyle/>{paras}')
    return shape(x0, y0, w0, h0, body, line="none", fill=None)


def shape(x0, y0, w0, h0, body="", line="none", fill="14283D", radius=False):
    sid = shape.next_id
    shape.next_id += 1
    geom = "roundRect" if radius else "rect"
    sppr = f'<p:spPr><a:xfrm><a:off x="{x(x0)}" y="{x(y0)}"/><a:ext cx="{x(w0)}" cy="{x(h0)}"/></a:xfrm><a:prstGeom prst="{geom}"><a:avLst/></a:prstGeom>'
    if fill:
        sppr += f'<a:solidFill>{color(fill)}</a:solidFill>'
    else:
        sppr += '<a:noFill/>'
    if line == "none":
        sppr += '<a:ln><a:noFill/></a:ln>'
    else:
        sppr += f'<a:ln w="12700"><a:solidFill>{color(line)}</a:solidFill></a:ln>'
    sppr += '</p:spPr>'
    return f'<p:sp><p:nvSpPr><p:cNvPr id="{sid}" name="Shape {sid}"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>{sppr}<p:txBody>{body or "<a:bodyPr/><a:lstStyle/>"}</p:txBody></p:sp>'


shape.next_id = 1


def line(x1, y1, x2, y2, fill="36C5C7", width=2):
    sid = shape.next_id
    shape.next_id += 1
    return (f'<p:cxnSp><p:nvCxnSpPr><p:cNvPr id="{sid}" name="Line {sid}"/><p:cNvCxnSpPr/><p:nvPr/></p:nvCxnSpPr>'
            f'<p:spPr><a:xfrm><a:off x="{x(x1)}" y="{x(y1)}"/><a:ext cx="{x(x2-x1)}" cy="{x(y2-y1)}"/></a:xfrm>'
            f'<a:prstGeom prst="line"><a:avLst/></a:prstGeom><a:ln w="{width*12700}"><a:solidFill>{color(fill)}</a:solidFill><a:tailEnd type="triangle"/></a:ln></p:spPr></p:cxnSp>' )


def slide_base(index, title, kicker="TRITON OPTIMIZATION AGENT"):
    shape.next_id = 1
    parts = [shape(0, 0, W, H, fill="091521"), shape(0.62, 0.52, 0.08, 0.68, fill="36C5C7")]
    parts.append(text_box(0.88, 0.48, 10.8, 0.25, paragraph(kicker, 9, "6D8AA0", True)))
    parts.append(text_box(0.88, 0.78, 11.5, 0.55, paragraph(title, 25, "F4F8FB", True)))
    parts.append(text_box(11.85, 0.53, 0.8, 0.28, paragraph(f"{index:02d} / 12", 10, "6D8AA0", True, "r")))
    parts.append(shape(0.88, 1.48, 11.55, 0.015, fill="1D3548"))
    return parts


def card(x0, y0, w0, h0, title, body, accent="36C5C7", body_size=14):
    parts = [shape(x0, y0, w0, h0, fill="102536", line="1D4054", radius=True)]
    parts.append(shape(x0, y0, 0.06, h0, fill=accent))
    parts.append(text_box(x0 + 0.2, y0 + 0.18, w0 - 0.34, 0.32, paragraph(title, 15, "F4F8FB", True)))
    parts.append(text_box(x0 + 0.2, y0 + 0.62, w0 - 0.34, h0 - 0.78, body if isinstance(body, list) else paragraph(body, body_size, "B7C9D8")))
    return parts


def bullets(items, size=14, fill="B7C9D8"):
    return [paragraph(item, size, fill, False, "l", True) for item in items]


def content(slide_parts, elements):
    slide_parts.extend(elements)
    return "".join(slide_parts)


def make_slides():
    slides = []
    p = [shape(0, 0, W, H, fill="091521")]
    p += [shape(0.72, 0.75, 0.1, 1.35, fill="36C5C7"), text_box(1.05, 0.77, 9.7, 0.3, paragraph("2026 HUAWEI BISHENG CUP", 11, "6D8AA0", True))]
    p += [text_box(1.05, 1.25, 10.8, 1.25, [paragraph("基于进化算法的", 34, "F4F8FB", True), paragraph("Triton 自动优化系统", 34, "F4F8FB", True)])]
    p += [text_box(1.08, 2.85, 8.8, 0.55, paragraph("面向昇腾 NPU 的自动算子优化方案", 19, "36C5C7", True)), text_box(1.08, 3.6, 6.4, 0.8, paragraph("让大模型负责提出优化候选，让确定性评测决定候选去留。", 16, "B7C9D8"))]
    p += [shape(8.55, 1.45, 3.7, 3.85, fill="102536", line="1D4054", radius=True)]
    for i, (t, c) in enumerate([("LLM", "36C5C7"), ("EVOLUTION", "F4A261"), ("REAL EVAL", "8BD3DD")]):
        y0 = 2.0 + i * 0.95
        p += [shape(9.0, y0, 2.8, 0.56, fill=c, radius=True), text_box(9.08, y0 + 0.12, 2.64, 0.25, paragraph(t, 13, "091521", True, "c"))]
        if i < 2:
            p.append(text_box(10.0, y0 + 0.62, 0.8, 0.24, paragraph("↓", 17, "6D8AA0", True, "c")))
    p += [text_box(1.08, 6.55, 5.5, 0.25, paragraph("团队：XXX  |  学校：XXX", 10, "6D8AA0"))]
    slides.append("".join(p))

    p = slide_base(2, "Triton 优化为什么需要自动搜索")
    p += card(0.9, 1.95, 3.55, 2.1, "性能依赖条件", bullets(["数据形状与 dtype", "访存、归约与布局", "Block Size 与 Warp 配置"], 13), "F4A261", 13)
    p += card(4.85, 1.95, 3.55, 2.1, "人工调优瓶颈", bullets(["依赖专家经验", "搜索空间难覆盖", "试错成本随算子数量增长"], 13), "36C5C7", 13)
    p += card(8.8, 1.95, 3.55, 2.1, "直接生成风险", bullets(["无法编译或改变语义", "公开样例过拟合", "失败候选重复出现"], 13), "D96C75", 13)
    p += shape(0.9, 4.65, 11.45, 1.25, fill="123044", line="2B7180", radius=True)
    p += text_box(1.18, 4.95, 10.9, 0.55, paragraph("核心问题：如何在固定时间、固定 Token 和真实硬件评测预算下，自动搜索出更多正确、有效且具有性能潜力的版本？", 18, "F4F8FB", True, "c"))
    slides.append("".join(p))

    p = slide_base(3, "比赛约束决定了系统的设计目标")
    p += card(0.9, 1.9, 2.7, 1.65, "20 分钟", "单个算子最长运行时间", "F4A261", 14)
    p += card(3.85, 1.9, 2.7, 1.65, "20 万 Token", "单个算子模型预算上限", "36C5C7", 14)
    p += card(6.8, 1.9, 2.7, 1.65, "Top-5", "统一接口最多返回 5 个版本", "8BD3DD", 14)
    p += card(9.75, 1.9, 2.7, 1.65, "正确性硬门槛", "功能通过后才进入性能评测", "D96C75", 14)
    p += shape(0.9, 4.15, 11.55, 1.55, fill="102536", line="1D4054", radius=True)
    p += text_box(1.25, 4.48, 10.85, 0.85, [paragraph("系统不是生成更多代码", 22, "F4F8FB", True, "c"), paragraph("而是在有限预算内，提高有效候选比例，持续提升 Top-5 中最佳有效结果。", 15, "36C5C7", True, "c")])
    slides.append("".join(p))

    p = slide_base(4, "总体方案：LLM 生成候选，进化算法管理搜索")
    steps = [("01", "基线算子", "读取 Triton 代码与约束"), ("02", "候选生成", "mutation / crossover"), ("03", "分层评测", "静态、正确性、性能"), ("04", "候选选择", "排序、去重、进入下一代")]
    for i, (n, t, b) in enumerate(steps):
        x0 = 0.95 + i * 3.05
        p += [shape(x0, 2.35, 2.35, 1.55, fill="102536", line="1D4054", radius=True), text_box(x0 + 0.2, 2.58, 0.55, 0.3, paragraph(n, 12, "36C5C7", True)), text_box(x0 + 0.2, 3.02, 1.95, 0.3, paragraph(t, 16, "F4F8FB", True)), text_box(x0 + 0.2, 3.4, 1.95, 0.25, paragraph(b, 11, "B7C9D8"))]
        if i < 3:
            p.append(text_box(x0 + 2.42, 2.92, 0.55, 0.35, paragraph("→", 24, "F4A261", True, "c")))
    p += [text_box(1.05, 4.85, 11.0, 0.55, paragraph("大模型只负责提出候选；静态检查、真实测试和 profiler 共同决定候选去留。", 18, "F4F8FB", True, "c"))]
    slides.append("".join(p))

    p = slide_base(5, "核心创新：从代码生成到证据驱动的自动优化")
    p += [shape(0.95, 2.0, 2.5, 2.8, fill="102536", line="2B7180", radius=True), text_box(1.2, 2.32, 2.0, 0.4, paragraph("LLM", 24, "36C5C7", True, "c")), text_box(1.2, 3.0, 2.0, 0.8, paragraph("提出结构化\n优化候选", 17, "F4F8FB", True, "c"))]
    p += [text_box(3.68, 3.0, 0.75, 0.5, paragraph("+", 28, "F4A261", True, "c"))]
    p += [shape(4.55, 2.0, 2.5, 2.8, fill="102536", line="2B7180", radius=True), text_box(4.8, 2.32, 2.0, 0.4, paragraph("EA", 24, "F4A261", True, "c")), text_box(4.8, 3.0, 2.0, 0.8, paragraph("组织候选\n持续进化", 17, "F4F8FB", True, "c"))]
    p += [text_box(7.28, 3.0, 0.75, 0.5, paragraph("+", 28, "F4A261", True, "c"))]
    p += [shape(8.15, 2.0, 3.7, 2.8, fill="102536", line="2B7180", radius=True), text_box(8.45, 2.32, 3.1, 0.4, paragraph("EVIDENCE", 21, "8BD3DD", True, "c")), text_box(8.45, 3.0, 3.1, 0.8, paragraph("失败反馈 +\n真实硬件验证", 17, "F4F8FB", True, "c"))]
    p += [text_box(1.0, 5.55, 11.4, 0.5, paragraph("四项创新：职责分工 · 失败知识 · 统一预算 · 全链路证据", 18, "36C5C7", True, "c"))]
    slides.append("".join(p))

    p = slide_base(6, "创新一：让大模型负责探索，让评测负责裁决")
    p += card(0.95, 1.95, 2.55, 2.85, "LLM", "理解算子\n提出候选\n生成 mutation / crossover", "36C5C7", 15)
    p += text_box(3.65, 3.05, 0.65, 0.35, paragraph("→", 24, "F4A261", True, "c"))
    p += card(4.35, 1.95, 2.55, 2.85, "进化算法", "管理种群\n选择父代\n调度下一代搜索", "F4A261", 15)
    p += text_box(7.05, 3.05, 0.65, 0.35, paragraph("→", 24, "F4A261", True, "c"))
    p += card(7.75, 1.95, 4.15, 2.85, "确定性评测", "静态检查：过滤明显无效代码\n正确性测试：验证计算语义\n真实硬件：判断性能收益", "8BD3DD", 15)
    p += [shape(0.95, 5.35, 10.95, 0.72, fill="123044", line="2B7180", radius=True), text_box(1.15, 5.55, 10.55, 0.25, paragraph("模型的生成能力 + 编译器与硬件的验证能力 = 更可靠的自动优化", 17, "F4F8FB", True, "c"))]
    slides.append("".join(p))

    p = slide_base(7, "创新二：失败结果也能推动下一代搜索")
    p += [shape(0.95, 2.05, 3.1, 2.7, fill="102536", line="D96C75", radius=True), text_box(1.25, 2.38, 2.5, 0.32, paragraph("失败候选", 18, "D96C75", True, "c")), text_box(1.25, 3.0, 2.5, 1.0, paragraph("编译失败\n正确性失败\n性能回退", 17, "F4F8FB", True, "c"))]
    p += [text_box(4.15, 3.0, 0.65, 0.35, paragraph("→", 24, "F4A261", True, "c"))]
    p += [shape(4.85, 2.05, 3.1, 2.7, fill="102536", line="F4A261", radius=True), text_box(5.15, 2.38, 2.5, 0.32, paragraph("结构化历史", 18, "F4A261", True, "c")), text_box(5.15, 3.0, 2.5, 1.0, paragraph("绑定算子、case、\n代码版本与环境", 17, "F4F8FB", True, "c"))]
    p += [text_box(8.05, 3.0, 0.65, 0.35, paragraph("→", 24, "F4A261", True, "c"))]
    p += [shape(8.75, 2.05, 3.1, 2.7, fill="102536", line="36C5C7", radius=True), text_box(9.05, 2.38, 2.5, 0.32, paragraph("搜索约束", 18, "36C5C7", True, "c")), text_box(9.05, 3.0, 2.5, 1.0, paragraph("减少重复试错\n识别高风险变换", 17, "F4F8FB", True, "c"))]
    p += [text_box(1.0, 5.45, 11.2, 0.45, paragraph("不只学习“什么有效”，也学习“什么在什么条件下无效”。", 18, "36C5C7", True, "c"))]
    slides.append("".join(p))

    p = slide_base(8, "创新三：在有限预算内获得更多有效候选")
    labels = [("TOKEN", "20 万"), ("TIME", "20 分钟"), ("EVAL", "真实测量"), ("OUTPUT", "Top-5")]
    for i, (a, b) in enumerate(labels):
        x0 = 0.95 + i * 3.0
        p += [shape(x0, 2.0, 2.45, 1.35, fill="102536", line="1D4054", radius=True), text_box(x0 + 0.18, 2.22, 2.1, 0.22, paragraph(a, 10, "6D8AA0", True, "c")), text_box(x0 + 0.18, 2.62, 2.1, 0.35, paragraph(b, 22, "F4F8FB", True, "c"))]
    p += [shape(0.95, 3.85, 11.35, 1.45, fill="123044", line="2B7180", radius=True), text_box(1.25, 4.15, 10.7, 0.75, paragraph("候选选择综合考虑正确性、性能、差异性、失败风险和评测成本。", 18, "F4F8FB", True, "c"))]
    p += [text_box(1.0, 5.7, 11.3, 0.35, paragraph("优化目标不是生成更多代码，而是在固定预算内获得更多有效候选。", 17, "F4A261", True, "c"))]
    slides.append("".join(p))

    p = slide_base(9, "创新四：从候选生成到性能结果的完整证据链")
    stages = [("父代", "ID / hash"), ("变异", "mutation kind"), ("验证", "compile / correct"), ("测量", "device / samples"), ("输出", "rank / Top-5")]
    for i, (a, b) in enumerate(stages):
        x0 = 0.85 + i * 2.45
        p += [shape(x0, 2.35, 1.85, 1.5, fill="102536", line="1D4054", radius=True), text_box(x0 + 0.12, 2.66, 1.6, 0.28, paragraph(a, 16, "F4F8FB", True, "c")), text_box(x0 + 0.12, 3.18, 1.6, 0.24, paragraph(b, 10, "36C5C7", True, "c"))]
        if i < 4:
            p.append(text_box(x0 + 1.88, 2.86, 0.52, 0.3, paragraph("→", 20, "F4A261", True, "c")))
    p += [shape(0.95, 4.65, 11.3, 1.0, fill="123044", line="2B7180", radius=True), text_box(1.15, 4.95, 10.9, 0.38, paragraph("每一个性能数字都能追溯到具体候选、具体测试和具体环境。", 18, "F4F8FB", True, "c"))]
    p += [text_box(1.0, 5.95, 11.3, 0.3, paragraph("本机结果、代理结果和官方 A2/A3 结果分别记录，避免证据越级。", 13, "B7C9D8", False, "c"))]
    slides.append("".join(p))

    p = slide_base(10, "多级评测：先保证正确，再追求性能")
    checks = [("语法", "Syntax"), ("接口", "Contract"), ("导入", "Import"), ("正确性", "Correctness"), ("性能", "Performance")]
    for i, (a, b) in enumerate(checks):
        x0 = 0.9 + i * 2.45
        p += [shape(x0, 2.35, 1.85, 1.35, fill="102536", line="36C5C7" if i < 4 else "F4A261", radius=True), text_box(x0 + 0.12, 2.62, 1.6, 0.25, paragraph(a, 16, "F4F8FB", True, "c")), text_box(x0 + 0.12, 3.05, 1.6, 0.2, paragraph(b, 9, "6D8AA0", True, "c"))]
        if i < 4:
            p.append(text_box(x0 + 1.88, 2.78, 0.52, 0.25, paragraph("→", 20, "F4A261", True, "c")))
    p += [text_box(1.0, 4.25, 11.3, 0.5, paragraph("任何一层失败，候选都会被记录并停止继续消耗更高成本的评测资源。", 17, "F4F8FB", True, "c"))]
    p += [shape(1.0, 5.25, 5.25, 0.68, fill="123044", line="2B7180", radius=True), text_box(1.18, 5.46, 4.9, 0.24, paragraph("通过候选 → 进入进化种群", 14, "36C5C7", True, "c")), shape(7.05, 5.25, 5.25, 0.68, fill="2C2027", line="D96C75", radius=True), text_box(7.23, 5.46, 4.9, 0.24, paragraph("失败候选 → 进入失败历史", 14, "D96C75", True, "c"))]
    slides.append("".join(p))

    p = slide_base(11, "本机 Ascend 开发验证：21 个公开算子资格闭环")
    p += [shape(0.95, 1.95, 2.75, 2.55, fill="102536", line="36C5C7", radius=True), text_box(1.2, 2.25, 2.25, 0.35, paragraph("21 / 21", 30, "36C5C7", True, "c")), text_box(1.2, 2.95, 2.25, 0.75, paragraph("当前可见 case\n本机资格矩阵", 16, "F4F8FB", True, "c"))]
    p += card(4.0, 1.95, 3.45, 2.55, "验证口径", bullets(["真实 Ascend 910B4", "正确性测试通过", "B,C,C,B 配对测量", "median ratio ≤ 1.03"], 13), "8BD3DD", 13)
    p += card(7.75, 1.95, 4.55, 2.55, "代表性 ratio", bullets(["chunk_cumsum：0.15686", "ep_gather：0.48065", "log_softmax：0.58038", "state_passing：0.85577"], 13), "F4A261", 13)
    p += [shape(0.95, 4.95, 11.35, 0.95, fill="2C2027", line="D96C75", radius=True), text_box(1.2, 5.2, 10.85, 0.35, paragraph("证据边界：这是本机 910B4、当前可见 case 的开发证据，不是官方 A2/A3 成绩，也不代表最终比赛排名。", 14, "F4F8FB", True, "c"))]
    slides.append("".join(p))

    p = slide_base(12, "总结：让 Triton 优化从人工试错走向自动搜索")
    p += [text_box(1.0, 1.95, 11.25, 0.65, paragraph("大模型提供搜索空间，进化算法组织搜索过程，真实评测提供可信反馈。", 22, "F4F8FB", True, "c"))]
    p += [shape(1.1, 3.05, 2.25, 1.25, fill="102536", line="36C5C7", radius=True), text_box(1.25, 3.42, 1.95, 0.3, paragraph("候选生成", 17, "36C5C7", True, "c")), text_box(3.55, 3.42, 0.7, 0.3, paragraph("→", 22, "F4A261", True, "c")), shape(4.35, 3.05, 2.25, 1.25, fill="102536", line="F4A261", radius=True), text_box(4.5, 3.42, 1.95, 0.3, paragraph("进化搜索", 17, "F4A261", True, "c")), text_box(6.8, 3.42, 0.7, 0.3, paragraph("→", 22, "F4A261", True, "c")), shape(7.6, 3.05, 2.25, 1.25, fill="102536", line="8BD3DD", radius=True), text_box(7.75, 3.42, 1.95, 0.3, paragraph("真实验证", 17, "8BD3DD", True, "c")), text_box(10.05, 3.42, 0.7, 0.3, paragraph("→", 22, "F4A261", True, "c")), shape(10.85, 3.05, 1.5, 1.25, fill="123044", line="36C5C7", radius=True), text_box(10.96, 3.42, 1.28, 0.3, paragraph("Top-5", 17, "36C5C7", True, "c"))]
    p += [shape(1.0, 5.2, 11.3, 0.78, fill="123044", line="2B7180", radius=True), text_box(1.2, 5.42, 10.9, 0.28, paragraph("在有限预算内，持续提出、验证、筛选并复用高质量 Triton 优化策略。", 17, "F4F8FB", True, "c"))]
    p += [text_box(1.0, 6.5, 4.0, 0.2, paragraph("谢谢各位评委", 11, "6D8AA0", True))]
    slides.append("".join(p))
    return slides


def xml_header(body):
    return '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>' + body


def slide_xml(body):
    return xml_header('<p:sld xmlns:a="%s" xmlns:r="%s" xmlns:p="%s"><p:cSld><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr/>' % (NS["a"], NS["r"], NS["p"]) + body + '</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sld>')


def package(slides):
    p_ns = NS["p"]
    rel_ns = "http://schemas.openxmlformats.org/package/2006/relationships"
    slide_ids = "".join(f'<p:sldId id="{255+i}" r:id="rId{i+2}"/>' for i in range(len(slides)))
    slide_rels = "".join(f'<Relationship Id="rId{i+2}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide{i+1}.xml"/>' for i in range(len(slides)))
    slide_overrides = "".join(f'<Override PartName="/ppt/slides/slide{i+1}.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>' for i in range(len(slides)))
    pres = xml_header(f'<p:presentation xmlns:a="{NS["a"]}" xmlns:r="{NS["r"]}" xmlns:p="{p_ns}"><p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId1"/></p:sldMasterIdLst><p:sldIdLst>{slide_ids}</p:sldIdLst><p:sldSz cx="{x(W)}" cy="{x(H)}" type="screen16x9"/><p:notesSz cx="{x(H)}" cy="{x(W)}"/><p:defaultTextStyle/></p:presentation>')
    pres_rels = xml_header(f'<Relationships xmlns="{rel_ns}"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="slideMasters/slideMaster1.xml"/>{slide_rels}</Relationships>')
    master = xml_header(f'<p:sldMaster xmlns:a="{NS["a"]}" xmlns:r="{NS["r"]}" xmlns:p="{p_ns}"><p:cSld><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr/></p:spTree></p:cSld><p:sldLayoutIdLst><p:sldLayoutId id="1" r:id="rId1"/></p:sldLayoutIdLst><p:txStyles><p:titleStyle/><p:bodyStyle/><p:otherStyle/></p:txStyles><p:clrMap accent1="36C5C7" accent2="F4A261" accent3="8BD3DD" accent4="D96C75" bg1="091521" tx1="F4F8FB"/></p:sldMaster>')
    layout = xml_header(f'<p:sldLayout xmlns:a="{NS["a"]}" xmlns:r="{NS["r"]}" xmlns:p="{p_ns}" type="blank"><p:cSld name="Blank"><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr/></p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sldLayout>')
    theme = xml_header(f'<a:theme xmlns:a="{NS["a"]}" name="Triton Agent"><a:themeElements><a:clrScheme name="Triton"><a:dk1><a:srgbClr val="091521"/></a:dk1><a:lt1><a:srgbClr val="F4F8FB"/></a:lt1><a:accent1><a:srgbClr val="36C5C7"/></a:accent1><a:accent2><a:srgbClr val="F4A261"/></a:accent2><a:accent3><a:srgbClr val="8BD3DD"/></a:accent3><a:accent4><a:srgbClr val="D96C75"/></a:accent4><a:accent5><a:srgbClr val="6D8AA0"/></a:accent5><a:accent6><a:srgbClr val="1D4054"/></a:accent6><a:hlink><a:srgbClr val="36C5C7"/></a:hlink><a:folHlink><a:srgbClr val="F4A261"/></a:folHlink></a:clrScheme><a:fontScheme name="Triton"><a:majorFont><a:latin typeface="Aptos Display"/><a:ea typeface="Microsoft YaHei"/></a:majorFont><a:minorFont><a:latin typeface="Aptos"/><a:ea typeface="Microsoft YaHei"/></a:minorFont></a:fontScheme><a:fmtScheme name="Triton"><a:fillStyleLst/><a:lnStyleLst/><a:effectStyleLst/><a:bgFillStyleLst/></a:fmtScheme></a:themeElements></a:theme>')
    ct = xml_header(f'<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/><Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"/><Override PartName="/ppt/slideLayouts/slideLayout1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"/><Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>{slide_overrides}</Types>')
    root_rels = xml_header(f'<Relationships xmlns="{rel_ns}"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/></Relationships>')
    master_rels = xml_header(f'<Relationships xmlns="{rel_ns}"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="../theme/theme1.xml"/></Relationships>')
    layout_rels = xml_header(f'<Relationships xmlns="{rel_ns}"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="../slideMasters/slideMaster1.xml"/></Relationships>')
    with ZipFile(OUT, "w", ZIP_DEFLATED) as z:
        files = {"[Content_Types].xml": ct, "_rels/.rels": root_rels, "ppt/presentation.xml": pres, "ppt/_rels/presentation.xml.rels": pres_rels, "ppt/slideMasters/slideMaster1.xml": master, "ppt/slideMasters/_rels/slideMaster1.xml.rels": master_rels, "ppt/slideLayouts/slideLayout1.xml": layout, "ppt/slideLayouts/_rels/slideLayout1.xml.rels": layout_rels, "ppt/theme/theme1.xml": theme}
        for i, body in enumerate(slides, 1):
            files[f"ppt/slides/slide{i}.xml"] = slide_xml(body)
            files[f"ppt/slides/_rels/slide{i}.xml.rels"] = xml_header(f'<Relationships xmlns="{rel_ns}"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/></Relationships>')
        for name, data in files.items():
            z.writestr(name, data)


if __name__ == "__main__":
    package(make_slides())
    print(OUT)
