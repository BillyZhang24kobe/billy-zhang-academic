---
# Documentation: https://wowchemy.com/docs/managing-content/

title: Understanding "Efficient Large-Scale Multi-Modal Classification"
subtitle: ""
summary: "This artical (in Chinese) analyzes a paper published in AAAI-18 about multi-modal classification, which is a hot topic back then."
authors: 
- admin
tags: []
categories: []
date: 2020-07-08T18:51:06Z
lastmod: 2020-07-08T18:51:06Z
featured: false
draft: false

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder.
# Focal points: Smart, Center, TopLeft, Top, TopRight, Left, Right, BottomLeft, Bottom, BottomRight.
image:
  caption: ""
  focal_point: ""
  preview_only: false

# Projects (optional).
#   Associate this post with one or more of your projects.
#   Simply enter your project's folder or file name without extension.
#   E.g. `projects = ["internal-project"]` references `content/project/deep-learning/index.md`.
#   Otherwise, set `projects = []`.
projects: []
---

### 内容简介和引子
多模态特征在自然语言处理领域的应用早在 [1] 中被提出。其中，语言信号和视觉信号是两种比较重要的模态信号，如何有效地结合这两种模态并且成功运用到NLP的任务中，是当时提出的一个非常重要的研究方向。在这之后，随着深度学习的崛起，如何在深度学习中融合多模态特征也是近几年非常受欢迎的一个领域。本篇文章主要解决的中心问题是“多模态分类”，其中涉及到了两种模态信息：文本和图像。在这个基础上，文章作者想要致力解决的一个问题是如何快速高效地将非常大规模的样本数据进行分类。值得一提的是，本文的作者是在NLP领域有着极其重要的推动作用的Tomas Mikolov，他在2013年提出了广为流传的Word2Vec 模型，为NLP的进一步发展奠定了强有力的基础。我个人认为，这篇文章提出的对于多模态信息融合的方法论，可以作为之后其他关于多模态融合的研究的强有力的理论支撑。

### 文章内容详解
下面对整篇文章的中心内容做一个简要的概括和描述。文章所聚焦的真实场景是“文本分类”，它对于很多其他的下游任务有着举足轻重的作用，比如：样本文档获取和划分，情感和主题的分类等。随着在日常生活中越来越多的多模态信息的出现，文本信息在非文本信息的作用下得到了加强，使得以多模态为核心的文本分类任务变得越来越重要。具体来说，本文探究的是使用了神经网络的多模态分类任务。其中，作者主要对两个研究问题做了探究：1）什么是融合多模态数据的最好的方法？2）如何能高效地完成多模态数据的融合？

在已有的多模态融合的方法论的基础上，本文作者对比和调研了能够加速模态融合过程的方法。特别地，对于离散的文本类型模态和连续的图片或视觉类型模态，作者提出了一个非常高效且新颖的方法论：将连续型的模态特征离散化。这样一来，训练变得更加快速，并且还能最小化内存损耗，使得对更加大规模的多模态分类成为了可能。

接下来，我们来从方法层面理解一下作者的做法。首先，作者用到了FastText 中文本分类的方法来作为baseline模型 [2]。为了让该baseline处理连续或者离散型的输入特征，作者做了如下处理：1）使用了从ResNet [3] 中训练得到的2048维的连续型特征；2）对于更大规模的数据集（比如：FlickrTag），作者采用了512维的特征；3）对于这些连续型的特征，作者采用了离散化的方式来提升模态融合的速度。为了让模型能够处理大规模的数据量，作者尽可能地让模型变得简单和高效，同时也能在baseline的基础上得到性能的提升。特别地，作者探究了不同的模型在多模态分类任务上的表现。为了对比，作者采用了相同的目标函数（如下图所示），最小化这个负平均对数概率。其中N代表了样本个数，o是神经网络的输出，x<sub>n</sub>是多模态的输入，y<sub>n</sub>是真实类别标注。

![](https://raw.githubusercontent.com/BillyZhang24kobe/PicGo/master/1.jpeg)

 这里简单介绍一下作者在实验中对比的几个模型:
   1. **Baselines（只考虑单模态特征）**
        - *文本模型 FastText*： 只涉及到了文本特征，完全忽略了视觉信号（visual signal）；
        - *连续型特征模型*：只涉及视觉信号（从ResNet中提取出的visual 特征），忽略文本特征。

   2. **连续型多模态模型（考虑多模态特征）** 
        - *相加型*：直接在特征向量层面对文本和视觉特征进行对应元素的相加
        - *最大池化型*：直接在特征向量层面对文本和视觉特征进行对应元素取最大值
        - *Gated 型*：对两种模态分别进行sigmoid 非线性操作，以此来计算一种模态相对于另一种模态的注意力（attention）。当然，这里具体对哪一个模态进行非线性操作是作为一个超参数来选择的；
        - *Bilinear 型*：用以抓取两种模态之间的任意联系。文章中作者直接加入了Gated 非线性操作，所以这个模型又叫做Bilinear-Gated。

   3. **离散化的多模态模型**
        - 为什么考虑离散化？
             连续型的多模态模型存在两个缺点：1）低效率的矩阵乘法； 2）在矩阵中存储大量的浮点型数据会耗费庞大的内存空间。因此，作者的解决方法是：对连续型的特征进行离散化的操作，以此将连续型的数据变成一个离散token所组成的序列。这样就能够在最基本的FastText初始阶段使用这些特殊的tokens，得到特殊的特征表示。这样就能弥补连续型多模态模型带来的缺陷。
             
        - 如何实现连续特征离散化？
             作者用到了Product Quantization，具体的流程是这样的：1）把一段连续型的特征向量分成大小相同的若干子向量；2）对每一个子向量进行k-means聚类操作；3）对于每一个图像，找到图像中每一个subwords对应的最近的cluster centroids，然后把找到的centroid与子向量的索引进行结合以获取离散化的向量（具体例子参照原文）。在进行离散化操作后，我们可以得到一个特殊tokens组成的序列。在这之后，这些tokens将作为新的"文本"输入FastText模型。此时的神经网络输出可以用以下式子表示：其中x<sub>n</sub><sup>d</sup>是离散化的特征，alpha是一个控制权重的超参数。除此之外，作者还介绍了一个新颖的quantization 方法 -- Random Sample Product Quantization (RSPQ)，为了保证重合的语义信息不被丢失。

         ![](https://raw.githubusercontent.com/BillyZhang24kobe/PicGo/master/2.jpeg)


在简单了解了文章作者的方法介绍之后，让我们来一起看看作者得到的结果是什么样的。

![](https://raw.githubusercontent.com/BillyZhang24kobe/PicGo/master/3.jpeg)

这个表统计了不同的模型在不同的数据集上的准确率。从中可以明显看出，不论哪一个数据集，基于多模态的分类模型（Continuous 和 Discretized）的准确率要远远超过传统的baselines模型（只考虑文本或者只考虑视觉特征）。由此可见，在文本分类任务中，当融合了多模态的特征之后，模型的性能往往会得到提升。

我们再来看作者对于模型训练速度做出的统计表（这里用到了大规模的训练集FlickrTag-1），对于连续型的多模态模型，可以注意到，Bilinear-gated 在三个数据集的准确率都是最高的。但是考虑到它的复杂度非常的高，训练时间很长，可以认为在偏好训练速度的前提下，Bilinear-Gated 模型并不是一个最佳选择。相反，一些很简单的模型（比如：相加型，最大池化型等）得到的准确度也非常的高，非常接近Bilinear-Gated模型。由此引申出的思考是：如果偏好准确率的话，可以选择Bilinear-gated 模型；如果对训练速度有要求，采用非常简单的相加型或者最大池化型模型也能带来很好地性能。同时，通过对连续型多模态模型进行离散化操作，可以将训练时间降到和FastText训练时间相仿的程度。这样带来的好处是：训练速度更快，并且准确率也能保证，达到了作者想要在大规模数据集上进行多模态分类的目标。

![](https://raw.githubusercontent.com/BillyZhang24kobe/PicGo/master/4.jpeg)

### 关于这篇文章的个人思考和总结
下面简单对这篇文章写一些自己的想法。文章的重要贡献体现在以下几个方面：1）对不同的多模态融合的方法进行比较和分析，从训练速度和模型准确率的层面对不同的方法进行了评估，以此得出了结论：用非常简单的模态融合方法（比如：相加或者最大池化）就能高效地达到非常好的模型表现；2）对连续型模态特征进行离散化处理，以此来进一步加速和简化整个模态融合过程；3）模型学习得到的离散化的特征表示能够为文本分类任务提供一定的可解释性。我个人非常喜欢作者对连续型特征的离散化处理，这样能将一些视觉特征转化为能够输入到FastText模型中的“文本特征”，这样就能充分利用FastText的高效学习的特点。同时，这样的创新能够使得在大规模数据上进行多模态分类成为可能。当然，我觉得这篇文章也存在一定的局限性：仅仅考虑了两种模态的信息（文本和视觉）。在之后的研究中也许能加入其他模态的信息（比如：音频）来辅助该分类任务。

总体来看，这篇文章对于工业界应该也可以提供非常多的启发。工业界可以提供非常多的数据，如何能有效地利用这些数据来帮助训练模型，一直是一个非常重要的问题。很多时候，工业界需要在准确度和速度上做出取舍，似乎是“鱼和熊掌不可兼得”的状态。但是这篇文章提供了一种很好的启发：如何能在保持相对高准确度的情况下还能快速训练模型。这需要对不同模态的特征进行观察和处理，选择最合理的模态融合方式。同时，对于分类任务，工业界也不能忽视可解释性带来的好处，这样才能在已有的模型架构基础上加以改进，达到更好的模型训练结果。