**Source:** https://yuxi-liu-wired.github.io/essays/posts/perceptron-controversy/

**Authors:** [[Liu, Yuxi]]

## Look Into
- **Marvin Minsky and Builder the robot**
- *Perceptron* by Marvin Minsky and Seymour Papert
- Definitions of connectionist vs symbolic AI
	- Symbolic AI: The founding metaphor of the symbolic system camp was that intelligence is symbolic manipulation using preprogrammed symbolic rules: logical inference, heuristic tree search, list processing, syntactic trees, and such. The symbolic camp was not strongly centered, though it had key players like Alan Turing, John McCarthy, Herbert Simon, and Marvin Minsky.
	- 
- Hebbian learning
- rheostats
- Hacker koan
- Duhem–Quine thesis
- Piaget constructivism
- On Society of Mind today: 
> Perhaps a modern reincarnation of such an idea would be the dream of Internet agents operating in a digital economy, populated by agents performing simple tasks like spam filtering, listening for the price of petroleum, etc. Some agents would interface with reality, while others would interface with agents. Some agents are organized at a higher level into DAOs, created by a small committee of simple “manager agents” serving as the interface and coordinators for other agents. DAOs can interface with other DAOs through little speaker-agents, which consist of a simple text filter for the torrent of information and then outsource to text-weaving agents to compose the actual messages they send out.

## Notes

**Minsky vs Rosenblatt**

Marvin Minsky, math PhD from Princeton
- Thesis on classical mathematical theory of McCulloch–Pitts neural networks
- Assigned a computer vision part of the "Robot plays ping pong" project to undergraduates as a summer project (might have forgotten to write in that part for the funding proposal)

Minsky and Papert - "symbolic AI" or Society of Mind thesis: "any brain, machine, or other thing that has a mind must be composed of smaller things that cannot think at all"
- This sounds like neural nets fall under the thesis but wait.
- Minsky attempts to build a robot, Builder, that can stack children's block using: a robot arm ("mechanical hand"), a camera ("television eye"), and software/computer. Many tiny programs are written for each part to work as a whole, leading Minsky to believe that human "ordinary common sense" is composed of millions of tiny processes.
- So to Minsky, AI needs to be composed of many tiny **symbolic programs.** "heterogeneous little separated components, not a homogeneous big connected piece."
- Neural nets do not follow this paradigm - each perceptron doesn't have to represent a specific subprocess of cognition, and a bunch of them have the same internal function, albeit with different parameter values after learning.

Frank Rosenblatt, psychology PhD from Cornell
> 'Rosenblatt irritated a lot of people;’ ‘Rosenblatt was given to steady and extravagant statements about the performance of his machine;’ ‘Rosenblatt was a press agent’s dream, a real medicine man'

Rosenblatt's "perceptron machine was a clever motor-driven potentiometer adaptive element that had been pioneered in the world’s first neurocomputer, the “SNARC”, which had been designed and built by Minsky several years earlier!"
- SNARC machine is a recurrent neural network that performs reinforcement learning by the Hebbian learning rule. 
- The machine was supposed to simulate a rat in the maze with a reward signal provided by an operator pressing a button.
- Networks and reward signal provided by basic circuitry like motors, clutches, capacitors 
- Without debugging, the machine simulated multiple rats, and Minsky saw that once one rat found a good path the other rats followed

Rosenblatt's perceptron technically had two layers, but the first layer were fixed 0-1 weights. This architecture was precisely illustrated and attacked in the *Perceptron* book.

Rosenblatt pushed to expand the single perceptron for speech recognition in the Tobermory prototype, making it multimodal. The Cornell project lasted 1961-1967.

***Perceptron* book didn't kill neural networks.**
- The book only looked at single perceptron
- Neural nets were already dead by the time the book came out in 1969 due to the lack of backprop and MLP
- Minsky and Papert had been criticizing perceptrons for some time

**That being said, the book does symbolize the dominance of symbolic AI and the start of the "AI winter" for neural networks.**
> In the middle nineteen-sixties, Papert and Minsky set out to kill the Perceptron, or, at least, to establish its limitations – a task that Minsky felt was a sort of social service they could perform for the artificial-intelligence community. (Bernstein 1981)

Stanford Research Institute (SRI) an early proponent of perceptrons alongside Roseblatt, re-focused on symbolic AI by 1969


**But neural nets/connectionism resurged in the 1980s, when MLP with backprop demonstrated counters to points made in *Perceptron*.**

**Minsky still objected to MLP-backprop trained on "theory-less toy data" - there is no way to prove the approach can be extended/generalized/scales beyond the experiment because the data can't be extrapolated.**

 - At Dartmouth 2006 AI Conference, "AI@50", Minsky said "You’re not working on the problem of general intelligence. You’re just working on applications."

**Papert had concerns that MLP-backprop prevents a society from developing different yet equally valid ways of knowing something.** 

Seymour Papert, math PhD
- Post doc under Jean Piaget who found that children build tiny models to explain reality and either modified or discarded them as they observed/learned more
- Papert believed in radical constructivism in children's education - rather than teach concepts according to an orthodox sequence of established by scientific consensus, there should be a way for children to learn by tinkering with the world around them by themselves (i.e., bricoleur, French from DIY-ers)
- With computers, this tinkering could even be unrestricted by the natural world.
- Papert hoped that computers could enable an epistemological pluralism - different ways of knowing
- Papert joined MIT in 1963 and collaborated with Minsky

**Minsky and Papert still hold to these objections as lessons of the original Perceptron publication, in spite of the MLP-backprop's demonstrating success**

"Was it because of the development of backpropagation, multilayer networks, and faster computers? Emphatically not. In fact, 1980s connectionists were not different from the 1960s connectionists. It is only the ignorance of history that made them think otherwise."

"There are no general algorithms and there are no general problems. There are only particular algorithm-problem pairs. "

A fully connected network at larger scales would get confused without receptive fields

Gradient descent can get trapped in local minima and at best works for a particular problem with particular minima

Toy data sets tend to have no sample noise, which works towards stochastic gradient descent's advantage

Rumelhart, Hinton, and Williams 1985 showed that several problems unsolvable by a single perceptron – XOR, parity, symmetry, etc – were solved by a two-layered neural network.
- Minsky and Papert noted the exponential increase in number of coefficients to get that to work and criticized that as evidence that neural nets for complex problems will have prohibitive computational costs

Papert: "If one were to ask whether any particular, homogeneous network could serve as a model for a brain, the answer (we claim) would be, clearly. No. But if we consider each such network as a possible model for a part of a brain, then those two overviews are complementary. This is why we see no reason to choose sides." 
 - Because yes, you can use neural nets for each part of a human brain or for different agents in a system, but then you'd be following Society of Minds thesis anyway.

Papert: "Massively parallel supercomputers do play an important role in the connectionist revival. But I see it as a cultural rather than a technical role, another example of a sustaining myth...I see connectionism’s relationship to biology in similar terms. Although its models use biological metaphors, they do not depend on technical findings in biology any more than they do on modern supercomputers"

Aside: I find it a bit funny that the guy who is against orthodox learning is promoting an orthodoxy in AI research.

**Curiously the XOR problem is commonly cited as a main reason why neural network idea failed.**
This isn't the most accurate way to tell the story. Any electrical engineer would have immediately thought of XOR as a test case at the time of 1960s and not bothered researching neural nets if they saw them fail XOR.

More accurately, the XOR problem could be solved with a two-layered network, but no one could come up with 

Liu: "When I first heard the story, I immediately saw why XOR was unsolvable by one perceptron, then took a few minutes to design a two-layered perceptron network that solved the XOR problem...If a high school student could bypass the XOR problem in a few minutes, how could it possibly have been news to the researchers in 1969?"

Liu: "If only we had tried an activation function, any activation function, other than the dreaded 0-1 activation function…"

Aside: The XOR problem is also not a matter of Big Data, GPUs, or even backpropagation algorithm.

**Liu argues that these objections have been disproven since 2019**
Jack Cowan: "What came across was the fact that you had to put some structure into the perceptron to get it to do anything, but there weren’t a lot of things it could do. The reason was that it didn’t have hidden units. It was clear that without hidden units, nothing important could be done, and they claimed that the problem of programming the hidden units was not solvable. \[Minsky and Papert] discouraged a lot of research and that was wrong."

Liu: "And so in 2012, Alex Krizhevsky cobbled together 8 GPUs and train a neural network that outperformed every symbolic or statistical AI.14 There are large homogeneous neural networks that work, and there are hints that some of them have small groups of neurons representing symbolic concepts, some of which are engaged in serial computation across the layers."

Liu: "Peering into any large enough software project, be it the Cyc project, or the Linux source code, one feels that it is easier to start anew than to add to it. Perhaps with thousands of years of very patient work and many evolutionary deadends, purely symbolic AI research can succeed in constructing a general intelligence in the elegant style sketched by Minsky. The irony is that symbolic programs do not scale while neural networks scale, the exact opposite of the lesson that Minsky and Papert wished to impart by their book."

"the final goal of general computer vision, capable of understanding real scenes, turns out to be far less about provably detecting edges and cubes and cones in a picture, and far more about having a large dataset. In this sense, it’s Minsky and Papert who were misled by their experiments with building block-playing robots in a block world. It’s their work that could not scale."
- You don't need to reliably solve sub-parts of a large problem in an elegant and theoretically generalizable way

Liu: "Several standard architectures constitute almost the entirety of neural networks nowadays – MLP, CNN, GNN, LSTM, VAE, and Transformers. Six is quite far from the thousands of architectures they explicitly predicted...Minsky and Papert hoped to show that there would be thousands of different problems, each requiring a bespoke algorithm implemented by a bespoke neural network. In this regard, their project has been fully debunked."