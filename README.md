# gpt.c

![gpt.c logo](https://github.com/MYusufY/gpt.c/blob/assets/gptdotc.png)

GPT version of [Andrej Karpathy](https://github.com/karpathy)'s [Llama2.c](https://github.com/karpathy/llama2.c).

Just like Llama2.c,

> This repo is a "fullstack" train + inference solution for ~~Llama2~~ GPT LLM,
> with focus on minimalism and simplicity.

And... it started as a weekend project for me just like Llama2.c was.

My vision, along with my small company, Tachion is making smaller machines more useful, and so reducing e-waste.

I already made (and will keep making) lots of projects with this vision. I can give [AgenticCore](https://agenticcore.tachion.tech/) as a recent example. But this time, i wanted to go further, for even smaller devices: microcontrollers.

Since i started making projects about AI, (was mostly training my custom [YoLo](https://pjreddie.com/darknet/yolo/) models, when i was ~10) i was mostly interested in AI-on-edge. But honestly i didnt think that we could go this far in a couple of years: LMs in microcontrollers.

At the moment i saw amazing Llama2.c project, i instantly thought implementing it into microcontrollers, especially ESP32's.

After a quick research, i realized [it was already achieved by humanity](https://github.com/DaveBben/esp32-llm)! :)

But i wanted to add my own twist, making a GPT-compatible version of Llama2.c project.

## Roadmap

 - [x] Make the GPT version of Llama2.c
 - [x]  Make a dataset generation automation program ([DataSeek](https://github.com/MYusufY/dataseek))
 - [x] Generate an instruct (QA) dataset with it, with microcontroller and daily speech knowledge.
 - [ ] Train a model around 250k-300k parameters using [train.py](train.py) with that custom dataset, and so make a small instruct LM for gpt.c.
 - [ ] Port gpt.c into ESP32, just like what [Dave Bennett](https://github.com/DaveBben) made but for [Llama2.c](https://github.com/karpathy/llama2.c).
 
... just to run a GPT-styled instruction-following language model on an ESP32!

# Repository Contents

- [train.py](train.py) -> training script
- [tokenizer.py](tokenizer.py) -> tokenizer generator
- [export.py](export.py) -> checkpoint converter/exporter script
- [model_gpt.py](model_gpt.py) -> GPT model architecture and configurations
- [run.c](run.c) -> inference script
- [Makefile](Makefile) -> makefile (• ᴗ •)

# Guide

## training

Unfortunately, **there's currently no available-to-download pretrained models,** so just for now, you should train your own model first and test it!

As i mentioned, i will keep working on this project, so i will be releasing multiple compatible models soon!

Here's how to train your own:
- Clone the repository
	``` bash
	git clone https://github.com/MYusufY/gpt.c.git
	cd gpt.c
	```
- Now, to train your model:
	```bash
	mkdir out
	python3 train.py
	```
	Again, for now, train.py has *[tinyshakespeare](https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt)* dataset  and all the configurations hardcoded in it. So it currently has no CLI usage, you just run it and it starts training. (Of course this is not practical, this is just a very early version!)
- Output
	When the training is completed, you should see your checkpoints saved under `out/` folder. Now, lets export it.
- Export / Convert the Checkpoint
	``` bash
	python3 export.py mymodel.bin --version 3 --checkpoint out/final_model.pt
	```
	Now, you should have mymodel.bin written.
	Almost done!

## run it!
- Generate tokenizer.bin
	Just like most of the other things, tokenizer.py is also in early stage, i cant say its the most advanced tokenizer script. But it still works, again hardcoded just for *[tinyshakespeare](https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt)* dataset :)
	``` bash
	python3 tokenizer.py
	```
- Let's run it!
	- Compile
		``` bash
		make run
		```
	- Run!
		Lets give "QUEEN" as an input because, you know, its Shakespeare!
		```bash
		./run mymodel.bin -i "QUEEN"
		```
		You should see the output!
		
# Benchmarks

Benchmarks will be released once the first models are trained and published!

# License

[MIT](LICENSE)
