import argparse
import os

import torch
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from transformers import AutoModelForCausalLM, get_linear_schedule_with_warmup, set_seed

from accelerate import Accelerator
from accelerate.utils import find_executable_batch_size
from data import get_dataloaders


def main(args):
    accelerator = Accelerator()

    peft_config = LoraConfig(
        bias="none",
        task_type="CAUSAL_LM",
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )

    set_seed(args.seed)

    with accelerator.main_process_first():
        if torch.cuda.is_available():
            model = AutoModelForCausalLM.from_pretrained(args.model_name)  # , torch_dtype=torch.float16)
        else:
            model = AutoModelForCausalLM.from_pretrained(args.model_name)

    # small fix to get gpt style generation working
    model.config.pad_token_id = model.config.eos_token_id

    model = get_peft_model(model, peft_config)

    # prints some nice information about the trainable parameters
    # and model stastics
    model.print_trainable_parameters()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    model, optimizer = accelerator.prepare(model, optimizer)
    starting_epoch = 0

    # we wrap the inner training loop into a function
    # in order to find the maximum batch size that fits into memory
    @find_executable_batch_size(starting_batch_size=args.batch_size)
    def training_loop(batch_size: int):
        nonlocal accelerator  # ensure they can be used in our context
        accelerator.free_memory()  # free lingering references

        train_dataloader, tokenizer = get_dataloaders(
            accelerator=accelerator,
            model_name=args.model_name,
            batch_size=batch_size,
            seq_len=args.seq_len,
        )

        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=0.06 * (len(train_dataloader) * args.num_epochs),
            num_training_steps=(len(train_dataloader) * args.num_epochs),
        )

        train_dataloader, lr_scheduler = accelerator.prepare(train_dataloader, lr_scheduler)

        # investigate if this is necessary
        # https://github.com/huggingface/accelerate/blob/main/examples/by_feature/automatic_gradient_accumulation.py
        gradient_accumulation_steps = 1
        if batch_size < args.batch_size:
            gradient_accumulation_steps = args.batch_size // batch_size
            accelerator.print(
                f"Batch size {args.batch_size} is too large for memory, reducing to {batch_size} and accumulating gradients over {gradient_accumulation_steps} steps"
            )

        for epoch in range(starting_epoch, args.num_epochs):
            model.train()
            total_loss = 0

            # if args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            #     # We need to skip steps until we reach the resumed step
            #     train_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
            #     overall_step += resume_step

            for step, batch in enumerate(train_dataloader):
                outputs = model(**batch)
                loss = outputs.loss / gradient_accumulation_steps
                total_loss += loss.detach().float()
                accelerator.backward(loss)
                if step % gradient_accumulation_steps == 0:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                if step % args.print_freq == 0:
                    accelerator.print(
                        f"[Training] Epoch: {epoch} | Step {step}/{len(train_dataloader)} - Loss: {outputs.loss:.2f} - LR: {optimizer.param_groups[0]['lr']:.7f}"
                    )

                if accelerator.is_main_process:
                    if step % 100 == 0:
                        model.eval()

                        with torch.no_grad():
                            with torch.autocast("cuda"):
                                x = "\r\n- Bachelor’s Degree in Computer Science or related degree\r\n- 4+ years professional experience in software development\r\n- Computer Science fundamentals in object-oriented design\r\n- Computer Science fundamentals in data structures\r\n- Computer Science fundamentals in algorithm design, problem solving, and complexity analysis\r\n- Proficiency in Java, C#, Scala, or other similar programming languages\r\n"
                                golden_sample = f"Write a modern and engaging job posting for the following basic qualifications: {x}\r\nResponse: \r\n"

                                inputs = tokenizer(golden_sample, return_tensors="pt")

                                input_ids = inputs["input_ids"]
                                if torch.cuda.is_available():
                                    input_ids = input_ids.to("cuda")

                                try:
                                    outputs = accelerator.unwrap_model(model).generate(
                                        input_ids=input_ids,
                                        do_sample=True,
                                        temperature=0.9,
                                        max_length=1024,
                                    )
                                    accelerator.print(
                                        tokenizer.batch_decode(
                                            outputs.detach().cpu().numpy(),
                                            skip_special_tokens=True,
                                        )
                                    )
                                except Exception as e:
                                    accelerator.print("Error while decoding golden_sample (expected)")
                                    accelerator.print(e)

                        model.train()

            adapter_weights = get_peft_model_state_dict(model=accelerator.unwrap_model(model))

            if args.output_dir:
                output_dir = os.path.join(args.output_dir, f"epoch_{epoch}")
                accelerator.save_state(output_dir)
                torch.save(adapter_weights, os.path.join(output_dir, "adapter.pth"))

            if args.model_dir:
                model_dir = os.path.join(args.model_dir, "model")
                accelerator.save_state(model_dir)
                torch.save(adapter_weights, os.path.join(model_dir, "adapter.pth"))

            accelerator.print(
                f"[Training] Epoch: {epoch} Completed | Loss: {(total_loss / (len(train_dataloader))):.2f} - LR: {optimizer.param_groups[0]['lr']:.7f}"
            )

    training_loop()

    # accelerator.end_training()
    accelerator.wait_for_everyone()


def parse_args():
    parser = argparse.ArgumentParser(description="Multilabel text classification")

    parser.add_argument("--model_name", default="EleutherAI/pythia-2.8b", type=str)

    parser.add_argument(
        "--batch_size",
        default=8,
        type=int,
        help="the training will find the max batch size that fits into memory and use gradient accumulation",
    )
    parser.add_argument("--num_epochs", default=30, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument("--seq_len", default=512, type=int, help="sequence length")
    parser.add_argument("--learning_rate", default=0.0005, type=float, help="initial learning rate")
    parser.add_argument("--seed", default=42, type=int, help="seed for initializing training. ")

    # LoRa
    parser.add_argument("--lora_r", default=8, type=int, help="LoRa r")
    parser.add_argument("--lora_alpha", default=16, type=int, help="LoRa alpha")
    parser.add_argument("--lora_dropout", default=0.1, type=float, help="LoRa dropout")

    parser.add_argument("--print_freq", default=10, type=int, help="print frequency")
    parser.add_argument("--output_dir", default=os.environ.get("SM_OUTPUT_DIR"))
    parser.add_argument("--model_dir", default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--resume", default="", help="resume from checkpoint")

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
