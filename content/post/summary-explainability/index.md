---
# Documentation: https://wowchemy.com/docs/managing-content/

title: Summary on "Generating Hierarchical Explaination on Text Classification via Feature Interaction Detection"
subtitle: ""
summary: This post (in Chinese) summarizes some interesting facts and thoughts about a paper regarding Interpretability in NLP, in the context of text classification.
tags: []
categories: []
date: 2020-07-14T18:39:37Z
lastmod: 2020-07-14T18:39:37Z
featured: false
draft: false

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder.
# Focal points: Smart, Center, TopLeft, Top, TopRight, Left, Right, BottomLeft, Bottom, BottomRight.
image:
  caption: ""
  focal_point: ""
  preview_only: false

authors:
- admin

# Projects (optional).
#   Associate this post with one or more of your projects.
#   Simply enter your project's folder or file name without extension.
#   E.g. `projects = ["internal-project"]` references `content/project/deep-learning/index.md`.
#   Otherwise, set `projects = []`.
projects: []
---

### 文章内容简介
深度学习的兴起被认为是人工智能崛起的第三次浪潮，它极大程度上推动了很多人工智能子领域的研究和进展，如：计算机视觉，自然语言处理等。然而，由于深度学习网络的架构非常的复杂，很多时候我们将其视为“Black Box（黑箱子）”。在一些任务中（如：文本分类，文本生成，图像识别等），我们已经能够训练出有着极高性能和准确度的模型。但是与此同时，这些优质模型的可解释性，一直是很多研究人员想要深入探索和学习的方向。因为只有当模型的结果具备良好的可解释性，我们才能充分地有理由依赖和信任模型的表现。

本文主要探究了在NLP中的文本分类模型的可解释性问题，通过识别不同输入特征（如：单词和短语）之间的联系(Feature Interaction Detection)，建立了层次化的解释(hierarchical explaination)，以此来观测输入特征中的词和短语是如何在不同层级上进行关联的。这种层级式的可视化能够帮助用户和研究者更加清晰地理解“黑箱”模型的决策（如在文本分类任务中对于不同类别的预测）。这篇文章中作者提出的通过生成Hierarchincal Explaination的方法，为模型可解释性的研究方向提供的强有力的实践支撑，能够有效促进该研究方向的进一步探索。

### 文章内容描述
本文作者以文本分类（具体来说是情感分类）为研究的主要背景，探究了如何通过识别输入特征（词语或短语）之间的关联，来解释分类器的最终表现和行为。具体来说，作者提出了一个"model-agnostic（与模型无关）" 的方法 -- HEDGE (Hierarchical Explaination via Divisive Generation)，通过递归地识别文本中特征词之间最弱的(weakest) 交互，然后基于此交互不断地把长文本片段分割成若干的小文本片段，构建了层级化的解释结构。如下图例子所示，HEDGE在输入的句子"a waste of good performance" 的基础上，生成了层级化的解释结构，并且能够在不同的层级中看出特征之间是如何进行交互的。
![](https://raw.githubusercontent.com/BillyZhang24kobe/PicGo/master/2-1.jpg)

下面，我们对文章作者提出的方法进行进一步解读。
- 算法层面: 作者提出了如何构建层次化解释(Hierarchical Explaination)。该算法的具体流程如下图所示。首先，输入部分(Input)的文本 x 由 n 个单词组成，同时y^代表某个模型输出的预测label。接下来，对于一个文本序列 x，我们可以将其划分成 P 个子文本序列（注：每个子文本序列彼此独立且无交集）。紧接着，HEDGE 可以对每一个子文本序列进行操作：
    - 找到分隔点 j；
    - 按照 j 对每个子文本序列进行分割。这样一来，每个子文本序列可以被进一步划分成两个更小的子文本序列；
    - 对得到的更小的子文本序列，评估其对于模型预测结果y^的贡献程度和影响(contributions)。

![](https://raw.githubusercontent.com/BillyZhang24kobe/PicGo/master/2-2.jpg)
由此算法可以看到，对于一个输入的文本序列，层级化的解释结构是通过一个自顶向下的过程来搭建的。在这个过程中，我们要回答两个问题：1）在下一个timestep，我们应该选择哪一个分割出来的文本子序列并且以此做下一步的分割？2）我们应该如何选择分隔点 j？幸运的是，这两个问题都可以通过下面的式子1来解决。这里的![Alt](https://raw.githubusercontent.com/BillyZhang24kobe/PicGo/master/2-4.jpg)表示的是两个子文本序列之间的交互分数 (interaction score)
![](https://raw.githubusercontent.com/BillyZhang24kobe/PicGo/master/2-3.jpg)

因为我们只聚焦探讨文章作者的核心算法和思路，具体的关于该分数的计算流程请参考原文章，这里我们不做过多赘述。从式子1中我们可以看出，内层的优化目标能够帮助我们找到合适的分割点 j，外层的优化目标则能帮助我们找到在下一个timestep 选择分割的子文本序列。同时需要注意的是，当我们每次完成步骤6和7（Algorithm1）之后，我们要把生成的 P 加入到层级表示 H 中，这样每次生成的层级都能够得到保存。算法1当中，每次迭代（步骤5-9）的最后一个步骤，就是评估新得到的子文本序列对于模型预测结果的贡献，然后更新贡献集合C。具体来说，作者定义出了一个可以量化每个特征（词或短语）对于模型预测结果的贡献度的式子（见式子5）。对于该式子的解读请参照原文section 3.3。到此，在进行完所有的迭代之后，算法1能够输出两个结果：1）贡献集合C_n-1: 包含了在每一个timestep生成的子文本序列和他们的贡献分数(importance scores)；2）层级H：包含每一个timestep 对于输入文本 x 的分割。基于这两个输出结果，一个完整的层级化解释结构就能通过可视化显现出来了。
![](https://raw.githubusercontent.com/BillyZhang24kobe/PicGo/master/2-5.jpg)

接下来，让我们一起看看作者得到的实验结果。为了展示不同模型在文本分类任务中的表现，作者挑选了三个典型的神经网络模型：LSTM，CNN 和 BERT 模型。特别地，文章用到了两个经典的数据集：[SST][1] 和 [IMDB][2]。Table 1 显示了这三个模型在不同数据集上的分类准确度。得到了初步的分类结果之后，作者对此分别进行了定量和定性的分析。
- 定量分析
  作者用了两个评价指标来衡量word-level 的解释：AOPC (Area Over the Perturbation Curve) 和 log-odds 分数。具体来说，AOPC越大，解释性越好；log-odds 分数越小，解释性越好。除此之外，作者还单独定义了另外的评价指标：cohesion-score，以此来评估给定子文本序列中词与词之间的交互程度。具体的分析结果如 Table 2 所示。可以看出，HEDGE 相比其他的方法要更能解释不同模型在两个数据集上的表现。
![](https://raw.githubusercontent.com/BillyZhang24kobe/PicGo/master/2-6.jpg)
![](https://raw.githubusercontent.com/BillyZhang24kobe/PicGo/master/2-7.jpg)
- 定性分析
  作者对于结果的定性分析主要体现在 HEDGE 相比于其他的方法，更能在词/短语层面以及它们之间的关联来进行对模型预测结果的解释。更多的对于定性分析的例子，请参照原文的附录部分.

从结果中，我们能够看出，HEDGE 不论是从定量还是定性的分析中，都能显现出非常有效的对于文本分类模型预测行为的解释能力。

### 自己的想法
总体上看，这篇文章的重要贡献主要体现在以下方面：
- 提出了一个自顶向下的可以通过识别特征交互，来创建层级化解释架构的方法。该方法能够对模型在任务中的表现提供合理有效的解释；
- 提出了一个简单且有效的能够定量特征对于模型结果的贡献的计分方法；

在行业和工业界，我觉得这篇文章能够给在探究模型可解释性的方向上提供很好的思路和启发。虽然在业界，很多时候算法工程师们不会过多地关注模型的可解释性，毕竟只要训练出来的模型能够在测试中拿到很好的成绩，就已经能够证明该模型的能力和表现，不会再深究为什么模型表现得那么好。但是反过来，一旦我们的模型表现没有达到预期，如果不对模型表现的可解释性进行探究，我们很难在现有的基础上对模型进行改进。所以我觉得，这篇文章很好的说明了可解释性对于理解模型的决策有着非常重要的作用。这不仅表现在对模型的改进上，还能让我们加深对于模型决策的自信和依赖程度。

<!-- ### 参考 -->
[1]: https://www.researchgate.net/publication/284039049_Recursive_deep_models_for_semantic_compositionality_over_a_sentiment_treebank
[2]: https://www.researchgate.net/publication/220873867_Learning_Word_Vectors_for_Sentiment_Analysis