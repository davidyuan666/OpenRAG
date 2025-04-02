# -*- coding = utf-8 -*-
# @time:2024/7/29 10:12
# Author:david yuan
# @File:agent_start.py
# @Software:VeSync



class AgentService(object):
    def __init__(self, *args, **kwargs):
        self.cfg = Config()
        self.agent_profile = None
        self.p_date = datetime.today().strftime('%Y%m%d')

    @staticmethod
    def parse_config(input_dict):
        cfg = Config()

        llm_name = input_dict.get("llm_name", "").lower()
        cfg.fast_llm_model = llm_name
        cfg.smart_llm_model = llm_name
        cfg.max_tokens_num = input_dict.get("max_tokens_num", 4096)
        if llm_name == "gpt-4":
            cfg.fast_llm_model = "gpt-3.5-turbo"

        return cfg

    @staticmethod
    def load_history(input_dict):
        history = input_dict.get("history", list())
        if not history:
            history = list()
        if isinstance(history, str):
            history = json.loads(history)
        return history

    def chat(self, input_dict):
        s = "============ INPUT_DICT ============\n"
        for key, val in input_dict.items():
            s += f"· {key.upper()}:\t{val}\n"
        print(s)

        chat_id = str(input_dict["id"])
        history = self.load_history(input_dict)
        self.cfg = self.parse_config(input_dict)
        self.agent_profile = AgentProfile(input_dict)

        print(self.cfg)
        print(self.agent_profile)

        try:
            agent = VagentLite(
                    cfg=self.cfg,
                    session_id=chat_id,
                    agent_profile=self.agent_profile,
                    lang=input_dict.get("lang", "en"))

            print("\033[95m\033[1m" + "\n***** Question *****" + "\033[0m\033[0m")
            print(input_dict["query"])

            agent_results = agent.chat(
                input_dict["query"],
                history=history)

            print("\033[95m\033[1m" + "\n***** Response *****" + "\033[0m\033[0m")
            print(agent_results["response"])

            result = {
                "id": chat_id,
                "response": agent_results["response"],
                "history": json.dumps(agent_results["history"], ensure_ascii=False),
                "chain_msg": agent_results["chain_msg"],
                "chain_msg_str": agent_results["chain_msg_str"],
                "more_info": agent_results["more_info"]
            }

        except KeyboardInterrupt:
            exit()
        except:
            print(traceback.format_exc())
            result = {
                "id": chat_id,
                "response": "error"
            }

        return result
