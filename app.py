import os
import dotenv
import time
import uvicorn

from openai import OpenAI
from sse_starlette import EventSourceResponse
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Union
from langdetect import detect

dotenv.load_dotenv()

DEFAULTS = {
    'HTTPX_TIMEOUT': 60,
    'TEMPERATURE': 0,
    'MAX_TOKENS': 4096
}

summary_refine_prompt_template = """\
Your job is to produce a final summary.
We have provided an existing summary up to a certain point: {answer}
We have the opportunity to refine the existing summary (only if needed) with some more context below.
------------
{text}
------------
Given the new context, refine the original summary.
If the context isn't useful, return the original summary.
The language of summary must keep in {language}.
"""

summary_prompt_template = """Write a concise summary of the following,
and the language of summary must keep in {language}.


"{text}"


CONCISE SUMMARY:"""

todo_refine_prompt_template = """\
Your job is to produce a final todo list.
We have provided an existing todo list up to a certain point: {answer}
We have the opportunity to refine the existing todo list (only if needed) with some more context below.
------------
{text}
------------
Given the new context, refine the original todo list.
If the context isn't useful, return the original todo list.
The language of todo list must keep in {language}.
"""

todo_prompt_template = """Write a concise todo list of the following,
and the language of todo list must keep in {language}:


"{text}"


CONCISE TODO LIST:"""


def get_env(key):
    return os.environ.get(key, DEFAULTS.get(key))


def make_todo_list(content: str, model_name: str):
    client = OpenAI()

    language = detect(content)
    length = len(content)
    chunk_size = 1500
    start_idx = 0
    end_idx = 0
    times = 1
    answer = None
    while end_idx < length:
        end_idx = start_idx + chunk_size
        if end_idx >= length:
            end_idx = length

        text = content[start_idx:end_idx]
        text_nolines = text.replace("\n", "\\n")
        print(f"idx=[{start_idx}, {end_idx}], text: {text_nolines}")
        start_idx = end_idx

        if times == 1:
            content = todo_prompt_template.format(text=text, language=language)
        else:
            content = todo_refine_prompt_template.format(answer=answer, text=text, language=language)

        messages = [{
            "role": "user",
            "content": content
        }]
        params = dict(
            messages=messages,
            stream=False,
            model=model_name,
            temperature=get_env("TEMPERATURE"),
            max_tokens=get_env("MAX_TOKENS"),
            timeout=get_env("HTTPX_TIMEOUT")
        )

        chat_completion = client.chat.completions.create(**params)
        answer = chat_completion.choices[0].message.content
        print(f"Todo times: {times}, answer: {answer}")
        times = times + 1

    return answer


def summarize(content: str, model_name: str):
    client = OpenAI()

    language = detect(content)
    length = len(content)
    chunk_size = 1500
    start_idx = 0
    end_idx = 0
    times = 1
    answer = None
    while end_idx < length:
        end_idx = start_idx + chunk_size
        if end_idx >= length:
            end_idx = length

        text = content[start_idx:end_idx]
        text_nolines = text.replace("\n", "\\n")
        print(f"idx=[{start_idx}, {end_idx}], text: {text_nolines}")
        start_idx = end_idx

        if times == 1:
            content = summary_prompt_template.format(text=text, language=language)
        else:
            content = summary_refine_prompt_template.format(answer=answer, text=text, language=language)

        messages = [{
            "role": "user",
            "content": content
        }]
        params = dict(
            messages=messages,
            stream=False,
            model=model_name,
            temperature=get_env("TEMPERATURE"),
            max_tokens=get_env("MAX_TOKENS"),
            timeout=get_env("HTTPX_TIMEOUT")
        )

        chat_completion = client.chat.completions.create(**params)
        answer = chat_completion.choices[0].message.content
        print(f"Summarize times: {times}, answer: {answer}")
        times = times + 1

    return answer


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class DeltaMessage(BaseModel):
    role: Optional[Literal["user", "assistant", "system"]] = None
    content: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_length: Optional[int] = None
    stream: Optional[bool] = False


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length"]


class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Optional[Literal["stop", "length"]]


class ChatCompletionResponse(BaseModel):
    model: str
    object: Literal["chat.completion", "chat.completion.chunk"]
    choices: List[Union[ChatCompletionResponseChoice,
                        ChatCompletionResponseStreamChoice]]
    created: Optional[int] = Field(default_factory=lambda: int(time.time()))


def predict(query: str, model_id: str):
    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(role="assistant"),
        finish_reason=None
    )
    chunk = ChatCompletionResponse(model=model_id, choices=[
        choice_data], object="chat.completion.chunk")
    yield "{}".format(chunk.json(exclude_unset=True))

    summary = summarize(query, model_id)
    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(content=f"Summary:\n {summary}", role="assistant"),
        finish_reason=None
    )
    chunk = ChatCompletionResponse(model=model_id, choices=[
        choice_data], object="chat.completion.chunk")
    yield "{}".format(chunk.json(exclude_unset=True))

    todo_list = make_todo_list(query, model_id)
    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(content=f"\n\nTodo List:\n {todo_list}", role="assistant"),
        finish_reason=None
    )
    chunk = ChatCompletionResponse(model=model_id, choices=[
        choice_data], object="chat.completion.chunk")
    yield "{}".format(chunk.json(exclude_unset=True))

    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(content=f"\n\nTranscription:\n {query}", role="assistant"),
        finish_reason=None
    )
    chunk = ChatCompletionResponse(model=model_id, choices=[
        choice_data], object="chat.completion.chunk")
    yield "{}".format(chunk.json(exclude_unset=True))

    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(),
        finish_reason="stop"
    )
    chunk = ChatCompletionResponse(model=model_id, choices=[
        choice_data], object="chat.completion.chunk")
    yield "{}".format(chunk.json(exclude_unset=True))
    yield '[DONE]'

test_content="""
为贯彻落实党中央关于在全党开展党纪学习教育的重大决策部署，深入学习修订的《中国共产党纪律处分条例》，中央党校（国家行政学院）官网官微从今日起开设《党纪学习教育问答》专栏，围绕习近平总书记关于全面加强党的纪律建设的重要论述、新修订《条例》的内容和要义，以及党员干部在实际工作、学习中常见的疑惑和违纪风险点等，邀请中央党校（国家行政学院）党的建设教研部专家学者进行解答，为广大党员干部在党纪学习教育中提供学习参考。
今天推出第一期《党纪学习教育的四个关键问题》，敬请广大党员干部持续关注。
党纪学习教育问答
第一期：党纪学习教育的四个关键问题
解答专家
祝灵君，中央党校（国家行政学院）党的建设教研部副主任、教授
1.从修订后的《条例》看，习近平总书记关于党的纪律建设有哪些新的要求？
答：一是贯彻全面加强党的纪律建设的总体要求。《条例》第一章体现了全面加强党的纪律建设的总体要求，即进一步明确制定条例的目的如增加保证“党的理论”的贯彻执行，进一步丰富党的纪律建设的指导思想如对“两个维护”作出最新表述、增加弘扬伟大建党精神、坚持自我革命要求、健全全面从严治党体系等内容，增加对党组织和党员遵守党章党规党纪的“四个自信”要求、践行正确权力观政绩观事业观要求，丰富纪律处分应当遵循的基本原则如增加执纪执法贯通的要求，深化运用监督执纪“四种形态”，由轻及重细化“红脸、出汗”第一种形态，确保第四种形态真正成为极少数，把严的要求贯彻到党规制定、党纪教育、执纪监督全过程，引导党员干部学纪、知纪、明纪、守纪，真正把纪律规矩转化为政治自觉、思想自觉、行动自觉。
二是释放越往后执纪越严的鲜明信号。全面从严治党核心是加强党的领导，基础在全面，关键在严，要害在治。关键在严，体现为严的基调、严的举措、严的氛围，就是真管真严、敢管敢严、长管长严。《条例》在第一章第四条明确增写“把严的基调、严的措施、严的氛围长期坚持下去”。宽是害、严是爱，严管本身就是厚爱，严管也要体现厚爱，这才是运用辩证思维、系统观念、治理理念管党治党建设党。如《条例》在第三章纪律处分运用规则中列举了从轻、减轻处分的情形，进一步细化监督执纪第一种形态，特别是将“三个区分开来”贯通于监督执纪全过程。
三是对各级各类党组织和全体党员学纪、知纪、明纪、守纪提出新的更高要求。2020年2月26日，习近平总书记在中央政治局常委会会议审议党委（党组）落实全面从严治党主体责任规定稿时指出：“目前，中央党内法规已经有几百部了，每部法规都提出了许多要求，但有的党员干部连通读都没有做到，甚至连执行的人都可能将其束之高阁，到了执行时就‘随手拈来’”。如此看来，各级党组织都要聚焦解决一些党员、干部对党规党纪不上心、不了解、不掌握等问题，组织党员特别是党员领导干部认真学习纪律处分条例，做到学纪、知纪、明纪、守纪，把遵规守纪刻印在心，内化为言行准则，不断强化纪律意识、增强纪律定力。
2.党的纪律建设在全面从严治党中的地位如何？
答：党的纪律是党的各级组织和全体党员必须遵守的行为规则，是维护党的团结统一、完成党的任务的保证。马克思主义政党相比其他任何政党都要重视严格的纪律约束和规范。1859年5月18日，马克思在《致恩格斯》一文中写道：“我们现在必须绝对保持党的纪律，否则将一事无成。”恩格斯在反驳巴枯宁无政府主义时指出：“没有任何党的纪律，没有任何力量在一点的集中，没有任何斗争的武器！”列宁格外强调新型无产阶级政党的严密纪律，提出党必须有“铁的纪律”，必须变成“一块整钢”。在革命战争年代，毛泽东提出：“加强纪律性，革命无不胜。”改革开放初期，邓小平指出：“谁也不能违反党章党纪，不管谁违反，都要受到纪律处分，也不许任何人干扰党纪的执行，不许任何违反党纪的人逍遥于纪律制裁之外。”2013年1月，习近平总书记在十八届中央纪委二次全会上指出：“我们党是靠革命理想和铁的纪律组织起来的马克思主义政党，纪律严明是党的光荣传统和独特优势。”
党的十八大后，习近平总书记多次分析党的领导和党的建设中存在的主要问题、深层次原因。比如，他指出：“我们当前主要的挑战还是党的领导弱化和组织涣散、纪律松弛。不改变这种局面，就会削弱党的执政能力，动摇党的执政基础，甚至会断送我们党和人民的美好未来。十八大之前有很多党内的同志和广大人民群众有所担忧，也就是在这里。”党的十八大将党的纪律建设写入党代会报告，党的十九大将党的纪律建设纳入党的建设总体布局，党的二十大作出全面加强党的纪律建设的战略部署。党的十八大以来，以习近平同志为核心的党中央把党的纪律建设摆在突出位置，作为全面从严治党治本之策，与时俱进推进理论、实践和制度创新，纪律建设成为新时代党的建设总体布局的新亮点。
3.如何借助《条例》学习贯彻全面从严治党要求？
答：《条例》既是执纪规则，也是行为规范。作为执纪规则，各级各类党组织、执纪专责机关必须严格执行党的纪律；作为行为规范，党员必须严格遵守、自觉接受党的纪律约束。《条例》贯通了全面从严治党要害在“治”的理念，即落实全面从严治党政治责任。
一是防止党员出现错误认识。表现为：动力不足，比如认为纪律教育“与自己工作不相关，学了用处也不大”；目的不明，比如认为“学规是为了不违纪，不违纪就不需要学规”；方法不当，比如坚持“囫囵吞枣，一知半解”；保障不强，比如党员学习难以做到全覆盖，党员受教育的优质师资力量不均衡；等等。为此，党员、干部首先要坚持对《条例》原原本本学，结合《习近平关于全面加强党的纪律建设论述摘编》学，结合习近平总书记系列讲话精神学，结合实际案例学，坚持集体学与自学相结合，坚持用心用情学。
二是提高党的纪律执行力。2014年1月14日，习近平总书记在十八届中央纪委三次全会上指出：“党的规矩，党组织和党员、干部必须遵照执行，不能搞特殊、有例外。各级党组织要敢抓敢管，使纪律真正成为带电的高压线。”各级党委、纪委监委应该严格执行党的纪律，既要正确运用“四种形态”、善于辨识“三个区分开来”，也要严格执纪标准和尺度，让全体党员、干部真正把学纪、知纪、明纪、守纪变成一种习惯、一种理念、一种情怀。
三是抓住领导干部这个“关键少数”。2016年1月12日，习近平总书记在十八届中央纪委六次全会上指出：“要养成纪律自觉，教育引导广大党员、干部特别是领导干部严格按党章标准要求自己，知边界、明底线，把他律要求转化为内在追求，自觉以身作则，发挥表率作用。”2023年，中办、国办印发《关于建立领导干部应知应会党内法规和国家法律清单制度的意见》，要求领导干部学习党内法规和国家法律。
四是加强组织领导。层层压实责任，如把党委（党组）抓党纪贯彻执行情况纳入巡视巡察和派驻监督重点；坚持问题导向，如对贯彻执行党纪不力的要批评教育、督促整改，严肃追责问责；加强监督执纪，如重点监督检查领导干部学习宣传和贯彻执行党纪情况；建立长效机制，如建立经常性党纪学习教育机制。
4.新修订的《条例》如何体现习近平总书记关于全面加强党的纪律建设的重要论述？
答：党的十八大以来，以习近平同志为核心的党中央深入阐明党的纪律建设的重要性、必要性，进一步明确党的纪律建设的概念、主体、原则和内容，形成了党的纪律建设丰富的理论成果、制度成果、实践成果。
一是在内容上，引入“党的规矩”概念。党的规矩是党的各级组织和全体党员必须遵守的行为规范和规则。2012年12月4日，党中央出台八项规定。2013年7月11日，习近平总书记在河北省平山县西柏坡九月会议旧址主持召开县乡村干部、老党员和群众代表座谈会，习近平总书记指出：“党的规矩、制度的建立和执行，有力推动了党的作风和纪律建设。”纪律是成文的规矩，一些未明文列入纪律的规矩是不成文的规矩；纪律是刚性的规矩，一些未明文列入纪律的规矩是自我约束的规矩。党的规矩包括党章这个总规矩、党的纪律的刚性约束、国家法律的硬规矩、党在长期实践中形成的优良传统和工作惯例。比如：党内不许搞团团伙伙，党内不允许不负责任地传播消息、发表议论，干部脱岗离岗需要向组织汇报，领导干部个人重大事项要向组织报告等等，都属于党的规矩。
二是在理念上，科学阐释纪法关系。党的十八大前，纪律处分中存在“以纪代刑”或“带着党籍蹲监狱”的现象，纪律处分条例许多规定与法律条文重复，导致出现了“违纪是小节，违法才去处理”的现象。2015年纪律处分条例修订就是要防止党员干部走向两个极端：要么是好同志、要么是阶下囚。习近平总书记在十八届中央纪委六次全会上指出：“无数案例证明，党员‘破法’，无不始于‘破纪’。只有把纪律挺在前面，坚持纪严于法、纪在法前，才能克服‘违纪只是小节、违法才去处理’的不正常状况，用纪律管住全体党员。”所谓纪严于法，体现在标准和措施上；所谓纪在法前，体现为把纪律处分挺在追究法律责任前面；所谓纪法分开，体现为纪法双守与纪法双施；所谓纪法贯通，体现为纪律审查与监察调查相贯通。
三是在体系上，注重党内法规制度“废、改、立”工作。建立健全党内法规制度体系，就是要立明规则、破潜规则，形成弘扬正气的大气候。党的十八大以来，中国共产党坚持以党章为根本遵循，建立健全党内法规制度体系，形成党章、准则、条例、规定、办法、规则、细则七类名称。立足“废、改、立”并举，党内法规制度体系不断完善，形成“1+4”党内法规制度体系，即党的组织法规、党的领导法规、党的自身建设法规制度、党的监督保障法规。截至2023年6月底，现行有效党内法规3802部，中央党内法规227部，部委党内法规190部，地方党内法规3385部。
四是在重心上，以政治纪律严起来带动各项纪律全面从严。党的纪律是多方面的，如政治纪律、组织纪律、廉洁纪律、工作纪律、群众纪律、生活纪律，但政治纪律是最重要、最根本、最关键的纪律。2013年1月22日，习近平总书记在十八届中央纪委二次全会上指出：“严明党的纪律，首要的就是严明政治纪律。”2015年10月8日，习近平总书记指出：“实际上你违反哪方面的纪律，最终都会侵蚀党的执政基础，说到底都是破坏党的政治纪律。因此，讲政治、遵守政治纪律和政治规矩永远排在首要位置。”作为党员、干部必须遵守的最高政治纪律，《条例》对“两个维护”作出新表述：“坚决维护习近平总书记党中央的核心、全党的核心地位，坚决维护以习近平同志为核心的党中央权威和集中统一领导。”
五是在尺度上，深化运用监督执纪“四种形态”。《条例》规定了监督执纪“四种形态”：“经常开展批评和自我批评，及时进行谈话提醒、批评教育、责令检查、诫勉，让‘红红脸、出出汗’成为常态；党纪轻处分、组织调整成为违纪处理的大多数；党纪重处分、重大职务调整的成为少数；严重违纪涉嫌犯罪追究刑事责任的成为极少数。”在“四种形态”中，属于党员在作风纪律方面苗头性、倾向性问题，或者违反党纪情节轻微的，可以运用第一种形态，不给予党纪处分。比如，《条例》第十七、十八、十九条规定了从轻、减轻或免予处分的情况。《条例》第十九条规定：党员行为虽然造成损失或者后果，但不是出于故意或者过失，而是由于不可抗力等原因所引起的，不追究党纪责任。《条例》附则第一百五十八条针对新旧条例适用突出了“从旧兼从轻”原则。
六是在主体上，进一步明确纪律处分权限。党的十九大通过的党章赋予党组相应纪律处分权限，2018年中办印发的《党组讨论和决定党员处分事项工作程序规定(试行)》规定：党组对其管理的党员干部实施党纪处分，应当按照规定程序经党组集体讨论决定，不允许任何个人或者少数人擅自决定和批准。党纪处分决定以党组名义作出并自党组讨论决定之日起生效。2022年9月，中共中央发布的《中国共产党处分违纪党员批准权限和程序规定》第六条规定：除本规定和有关党内法规另有规定外，给予各级党委管理的党员警告、严重警告处分，可以由同级纪委审查批准；给予其撤销党内职务、留党察看或者开除党籍处分，须经同级纪委审查同意后报请这一级党委审议批准。
"""

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    if request.messages[-1].role != "user":
        raise HTTPException(status_code=400, detail="Invalid request")
    user_content = request.messages[-1].content
    #user_content = test_content
    generate = predict(user_content, request.model)
    return EventSourceResponse(generate, media_type="text/event-stream")


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8001)
