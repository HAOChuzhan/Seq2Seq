import re
import random
from utils.tools import word_count

delivery_pattern = re.compile(
    r"收到|"
    r"我的快递|"
    r"收货|"
    r"商品|"
    r"发货|"
    r"物流|"
    r"快递"
)

def intent_update(msg, dialog_status):
    if delivery_pattern.search(msg):
        dialog_status.intent = "delivery"
    elif dialog_status.intent == "delivery":
        dialog_status.intent = None
    return dialog_status

def delivery_handle(msg, dialog_status):
    if re.search("商品|我的东西|我的快递", msg):
        if re.search("破损|损坏", msg):
            return "您好，一旦商品包装出现破损，请咨询下配送师傅是否可以查看商品是否完好，如商品完好，请您放心签收，如商品破损，请您拒收商品；如已签收，请您在签收24小时内提交服务单并上传图片，我们会尽快为您处理"
        if re.search("发错", msg):
            return "小店自营商品：若您确认收到的商品不是您订单中订购的商品，可直接点击“客户服务”下的返修/退换货或商品右侧的申请返修/退换货，出现返修及退换货首页，点击“申请”即可操作换货。第三方商家商品：请直接联系商家的在线客服帮助您退/换货。"
        if re.search("送达|送到|到达|达到", msg):
            if re.search("没有|未", msg):
                return "您好，您可以登陆电脑端的商城首页—点击右上方“客户服务”—进入帮助中心首页—选择常用自助服务【我要催单】"
            else:
                return  "您可以进入“我的订单”点击对应订单查询即可看到物流更新。"
    if re.search("收货|收东西",msg):
        if re.search("不方便|不太方便",msg):
            return "您好，若您暂不方便收货，配送员在配送时电话联系您的时候，你可与配送员协商您的需求，配送员会尽量为您安排。"
    
    if re.search("物流|快递",msg):
        if re.search("查询|查看",msg):
            if re.search("如何|怎么|帮|给",msg):return "您可以进入“我的订单”点击对应订单查询即可看到物流更新。"
        
        if re.search("到哪|到那",msg):return "您可以进入“我的订单”点击对应订单查询即可看到物流更新。"
        
        if re.search("没有更新|[未不]更新",msg):
            return "您好，下单后，您的订单中物流信息长时间未更新，有可能是系统问题，或者是订单出现了异常，请您直接联系在线客服人员处理。"
    
        if re.search("更新错",msg):
            return "第三方卖家商品订单物流信息若更新错误，请直接与商家客服取得联系。"
        
        if re.search("派送|签收",msg):
            if re.search("未送[达到]|没有送[到达]",msg):
                return "您好，您可以直接电话联系配送人员。"
        
        if re.search("送达|送到|到达|达到",msg):
            if re.search("没有|未",msg):
                return "您好，您可以直接电话联系配送人员。"