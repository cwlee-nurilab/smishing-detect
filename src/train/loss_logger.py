from transformers import TrainerCallback


class LossLoggerCallback(TrainerCallback):
    def __init__(self, log_path):
        self.log_path = log_path

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return

        with open(self.log_path, "a", encoding="utf-8") as f:
            step = logs.get("step", state.global_step)

            if "loss" in logs:
                f.write(
                    f"[TRAIN] step={step} "
                    f"loss={logs['loss']:.6f} "
                    f"lr={logs.get('learning_rate', 0):.8f} "
                    f"epoch={logs.get('epoch', 0):.4f}\n"
                )

            if "eval_loss" in logs:
                f.write(
                    f"[EVAL] step={step} "
                    f"loss={logs['eval_loss']:.6f} "
                    f"f1={logs.get('eval_f1', 0):.6f} "
                    f"accuracy={logs.get('eval_accuracy', 0):.6f}\n"
                )