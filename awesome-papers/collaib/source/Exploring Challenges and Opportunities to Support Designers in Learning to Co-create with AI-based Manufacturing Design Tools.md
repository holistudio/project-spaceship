
**Link:** https://arxiv.org/pdf/2303.00192

**Authors**: [[Gmeiner, Frederic]], [[Nikolas Martelaro]]

## Questions
- What if AI can infer design goals rather than have users specify them? 
- Strategies of human-human collaboration may not readily translate into strategies or even just expectations of human-AI collaboration, because people still expect different things like levels of knowledge or competency in AI than they do humans. If anything, human-human collab informs some requirements and goals of human-AI, but not strategies how to achieve them.
- What's the best way to address the tiny holes issue? A better algorithm in the black box that fixes them automatically? A user annotation and AI tries to fix that?
-  What if generative AI is best learned in pairs of humans? Can additional UI changes really be as good as having a human guide?
- But do people really want Clippy?
	- Not all interruptions are appreciated, especially during flow state.
	- 

## Look Into
 - Effective Human-human collaborations grounding in communication shared mental models (citations 5,6,10,31,40,77,85)
 - Citation 43 - genAI for consumer goods
 - Citation 60 - building layouts
 - 81 - what role makes expect for involving AI in digital fab
 - 55, 70 - helping users refine mental models of AI
 - 35 - shared mental models
 - 53 - coordination of actions
 - 26, 62 - "intelligent tutoring systems can detect and respond to student hedging"
 - 67 - "spatial-action language" terminology describing what happens during design critiques

## Problem

Generative AI tools that take more agency in design are considered "co-creators" but do not usually work well collaborating with human users.

But generative AI's assistance is necessary for design tasks with high complexity.

Part of the challenge in collaborating is that users do not directly manipulate the design geometry/objects themselves but specify design goals and constraints, and then have to deal with whatever the generative AI comes up with. 

### Specific Challenges

 - The black box of AI makes it difficult for users to have a shared mental model of the AI thinks.
 - **Learnability of AI:** There have been studies on making genAI possible and making interfaces for users to interact with them, but not a lot of studies on how users **learn to use** the genAI tool.

## Related Works

 - Effective Human-human collaborations
	 - 
 - AI based design tools - making GenAI possible and making interfaces for users to interact with them, and studying how usable the interfaces might be, but don't focus on how users learn to use them.
 - Learning complex software - tutorials/widgets/self-directed learning processes have been studies for very complicated software in general. But once GenAI is in the mix, the process of learning to use the same software can change and this is not well examined.
 - Human-human collaborations - it remains an open question of how much human-AI collaboration should mirror
	 - grounding in communication
	 - theory of mind - ability for people to be aware of their own and others beliefs/intentions/perspectives
	 - shared mental models (citations 5,6,10,31,40,77,85, 35)
 - Team learning - coordination of actions, negotiation, constructive conflict

## Approach

Inform the design of human-AI collaboration processes in generative AI tools by examining:
1. How new users to generative AI tools learn to get better at using them OR struggle to do so. (RQ1a and b)
2. How human-to-human collaboration works and in what ways can that inform the human-AI collaboration in generative AI tools? (RQ2)
3. What needs, expectations, and strategies should inform generative AI tools for better human-AI collaboration? (RQ3)

**RQ1** is studied via think-aloud sessions in Study 1. Individual users try to learn while researchers observe
- **Mechanical design task:** Engine bracket with Fusion360, solver produces different design options and users pick or modify design constraints (loads, obstacles). Generation can take a few hours
- **Industrial design task:** Bottle holder on a bike using Simulearn (Rhino3D plugin), users can change the grid model/actuator ratio just simulate (human design)  or interact with the design and see what the AI does to re-optimize (human-AI) or make a target shape and have AI  automatically come up with something (AI design). Generation takes a few seconds/minutes
- Participants (n=7 for each task) used the basic software before but not the GenAI - expected to not struggle with basic UI but were new the GenAI part of the tool and had to learn to use it.
- Pilot study with experts doing the same task made sure the task was sufficiently complex but not overwhelming, could be done in multiple sessions

1. Intro session: 30 min intro and users start using for the task with researcher observation
2. Homework session: Users continue working while thinking out loud no time limit, multiple sessions
3. Interview

**Analysis methods:** Evaluation of design outcomes in terms of either design objectives (mechanical design) or human designer satisfaction with final design, video interaction analysis, reflexive thematic analysis

**RQ2** is studied in Study 2 by having a human user use the same GenAI tool with an experienced guide. Researchers just observed natural interactions
- Guides were recruited from Study 1
- *This reminds me of paired programming for learning software*

1. Intro session: 30 min intro and users start using for the task with researcher observation
2. Homework session: Users continue working while thinking out loud, 50 minute sessions
3. Interview

**RQ3** is addressed using the observations and analysis from studies 1 and 2


## Key Findings

### Study 1 - Think Aloud while using GenAI tool

- Users generally valued AI assistance but struggled in learning to co-design and interpret why the GenAI was generating certain geometries
- Users that learned quickly to user GenAI systematically tested the boundaries of capabilities and came up with their own explanations on why it was generating certain geometries, and readily reflected on what was happening inside that Gen AI black box
- Only 1 participant of 7 was fully satisfied with Mechanical Design and met the structural requirements

**Those who learn, do**
- Users that learned, systematically experimented with extreme grids or constraints to get a better mental model of AI behavior
- Users reflected by sketching and explaining how forces were acting by themselves

**Challenges using GenAI**
- No one thought the tasks were too difficult themselves but found the generative AI parts unintuitive
- Mechanical design participants failed to sometimes specify the correct structural loads and constraints
	- But this could be a major issue with Fusio360 in general, not specific to the generative AI part.
	- *Seems a little uncertain whether some issues were due to basic UI rather than generative AI. Moreover, it's possible that users will blame the generative AI for mistakes that result from the basic UI*
- Industrial design participants opted for the human design mode after giving up using the AI and human-AI hybrid modes.
- Artifacts like tiny holes or super thin or thick parts particularly confused the mechanical designers
- "Am I causing the problem or is the AI doing that?" Hard to trace back parameter influences on final result
- "I felt SimuLearn had more control over it than I did"
- Users tried some hacky work arounds but then "I was making things to satisfy the software instead of it kind of adapting to my needs"
- *Seems like users also expect AI to check for fabrication issues down stream rather than just meeting structural load requirements*

### Study 2 - Learning with a Peer

Reflexive thematic analysis
- Guide responds to designer's actions, verbalized confusion, or requests for help with step by step instructions
	- *Yeah, it must be responsive to struggles observed, NOT a tutorial when I open the software*
- Guide triggered moments of reflection with questions like "Is that what you envisioned?"
- Guides had the systematic experiments to test extremes and help users develop mental models
- Guides used mouse cursor and other nonverbal gestures to communicate
	- *Ok this is common even for complex software, not specific to AI*

### Needs and Expectations of AI
- AI should have context awareness to proactively guess at what the designer is trying to do and offer suggestions
	- *But do people really want Clippy?*
- AI should anticipate real-world design issues and remind users of fundamental engineering principles
- AI should be conversational

### Design Opportunities for Better Human-AI collaboration
Multimodal feedback and communication from the AI is a must as this illustrates and explains fundamental principles, communicates concerns and recommendations, feedback

Multimodal feedback from the users as well - annotations of observed GenAI design flaws especially

AI should detect "designers 'level of certainty"
- *Struggle is real detection*


## Takeaways

 - It remains an open question of how much human-AI collaboration should mirror human-human collaboration. This study identifies areas of improvement but did not develop any tools of their own to demonstrate or test their human-AI collaboration needs/expectations/strategies 
	 - This seems particularly needed given the discussion of team learning (negotiation, coordination) that is clearly missing from the GenAI tools studied but remains unclear how this process should work for human-AI collaboration
	 - And these expectations from users depends on how open each user is to the idea of having an generative AI help out in the first place.
 - It's also still an open question of how a tool (maybe with AI) can just detect that a user is struggling to use it, the same way a human guide does.
 - it's possible that users will blame the generative AI for mistakes that result from the basic UI. This makes evaluating generative AI's intuitiveness a bit tricky.





