import os
from task_dialog import start_task
from task_dialog import delivery_task
from task_dialog import finish_task
from task_dialog import short_query_task
from task_dialog import invoice_task
class TaskCore(object):

    intent_update_func = [
        ("start","start_task.intent_update"),
        ("delivery", "delivery_task.intent_update"),
        ("short_query", "short_query_task.intent_update"),
        ("invoice", "invoice_task.intent_update"),

        ("finish", "finish_task.intent_update"), 
    ]

    intent_not_reset = set(["sale_return", "refund", "invoice", "unbind", "price_protect"])

    intent_handle_func = {
        "start": "start_task.start_handle",
        "delivery": "delivery_task.delivery_handle",
        "short_query": "short_query_task.short_query_handle",
        "invoice": "invoice_task.invoice_handle",

        "finish": "finish_task.finish_handle",
    }
    # @classmethod
    # def _slots_update(msg, dialog_status):


    @classmethod
    def task_handle(cls, msg, dialog_status):
        try:
            response = None

            if dialog_status.intent not in cls.intent_not_reset:
                dialog_status.intent = None
            # 利用正则依次判断该句子的意图
            for intent, update_func in cls.intent_update_func:
                # dialog_status = cls._slots_update(msg, dialog_status)
                dialog_status = eval(update_func)(msg, dialog_status)
                print(dialog_status.intent)
                # 句子意图识别对应，执行对应的回复处理
                if dialog_status.intent == intent:
                    handle_func = cls.intent_handle_func[dialog_status.intent]
                    response = eval(handle_func)(msg, dialog_status)
                    if response:
                        break
            print("[DEBUG] intent = %s, task_response = %s" % (dialog_status.intent, response))
            return response, dialog_status
        except Exception as e:
            print("[DEBUG] msg = %s, error = %s" % (msg, e))
            return response, dialog_status
