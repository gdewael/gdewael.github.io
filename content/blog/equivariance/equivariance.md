---
title: Visualizing equivariances in transformer neural networks
description: This blogpost visualizes geometric priors in transformer neural networks
date: 2024-05-20
tags:
  - research
---

Transformer neural networks have become the dominant architecture within many subfields of deep learning.
Their success is partly owed due to the fact that self-attention is a very generic operation in terms of the geometric priors it uses[^geom].
The following blogpost interactively visualizes some geometric priors in transformers.
The target audience for this blogpost are those who are already (at least vaguely) familiar with self-attention and want to see some simple visualizations of what positional encodings do to them[^illustrated].

As a quick prerequisite, let us recap the self-attention formula via a simple example.
Consider the sentence: "Love conquers all".
Each word in this sentence can be assigned an embedding vector, which may look like the following:

<div style="text-align: center;" id="vis-x"></div>

These three vectors gives us an input matrix $\boldsymbol{X} \in \mathbb{R}^{n \times d}$ consisting of $n=3$ input elements, each with $d=2$ (hidden) features.
To perform self-attention on this input matrix $\boldsymbol{X}$, one first takes three linear transformations of $\boldsymbol{X}$ with learned weights $\boldsymbol{W}_{i \in \{q,k,v\}} \in \mathbb{R}^{d \times d}$. Visualized:

<div style="text-align: center;" id="vis-qkv"></div>


Note that I have chosen simple word embeddings and projections to simplify following along with computations.
In practice, these weights are learned.

After projection, self-attention is performed as follows:

$$
\boldsymbol{Z} = \mathrm{softmax}\left(\frac{\boldsymbol{Q} \boldsymbol{K}^\top}{\sqrt{d}}\right) \boldsymbol{V}
$$

With $\mathrm{softmax}\left(\frac{\boldsymbol{Q} \boldsymbol{K}^\top}{\sqrt{d}}\right)$ often described as the attention matrix $\boldsymbol{A} \in \mathbb{R}^{n \times n}$. Denoting $f(\cdot) =\mathrm{softmax}\left(\frac{\cdot}{\sqrt{d}}\right)$ as a normalizing function, one may re-write the whole operation as:

$$
\boldsymbol{Z} = f\Big(\boldsymbol{X}\boldsymbol{W}_q (\boldsymbol{X}\boldsymbol{W}_k)^\top\Big)\boldsymbol{X}\boldsymbol{W}_v
$$

Via this equation, one sees that, intuitively, self-attention is nothing more than three projections of $\boldsymbol{X}$ multiplied with eachother, with a normalization function in between. Self-attention may be visualized as:

<div style="text-align: center;" id="vis-attn"></div>
<p style="text-align: center;">
  <i><small>(Hover over the elements highlighted in blue to see computation)</small></i>
</p>

For the purpose of the visualizations, I've conveniently ignored the multiple heads that are typically used with self-attention.

## Permutation equivariance in self-attention

The words in the previous example sentence "Love conquers all" can be scrambled in a number of ways, and still form a correct sentence, e.g. "All love conquers".
We can play around with the previous visualizations of self-attention by adding a shuffle button:

<div style="text-align: center;" id="vis-shuffle"></div>

If you play around with the shuffling, you will notice that, if elements of $\boldsymbol{X}$ are reordered, the output $\boldsymbol{Z}$ is similarly reordered without otherwise changing.
Within the framework of geometric deep learning, this property is called *permutation equivariance*.
More formally, with the self-attention function denoted as $f_\text{attn}$, we can see that $f_\text{attn}(g(\boldsymbol{X})) = g(f_\text{attn}(\boldsymbol{X}))$, for any shuffling operation $g$.

Permutation equivariance is useful for any kind of data modality where the inputs are not really a sequence, but can rather be described as a set (i.e. ordering does not matter).
In language, however, it does, which is why transformers were originally proposed along with positional encodings.

## Positional embeddings break permutation equivariance

Positional encodings are most-simply introduced by adding a position-dependent signal to the input.
For example:

<div style="text-align: center;" id="vis-pos"></div>

You will see that permutation equivariance is broken by the positional encodings.
I.e., a shuffled input will not return the same output - albeit shuffled the same way - anymore.
By communicating positional indices, we do not operate on an unordered set.
Rather, the model becomes a true sequence model.

Note that this example features positional encodings that simply contain the positional indices.
In practice, positional encodings may be sinusoidal (which has a nice decaying similarity effect on the dot product attention matrix), as in the original transformer publication[^vaswani].
Given enough data, one may also choose to learn the positional embeddings from scratch, as in the BERT model[^bert].

## Time-shift (translation) equivariance through relative positional encodings

In many data domains, the absolute positioning of elements in the sequence does not matter.
In these domains, how signals co-occur relative to eachother may be more relevant.
For example, in images, the absolute location of a cat's ear is inconsequential to its detection.
Rather, the fact that a cat's ear should be located on top of its head is a relevant signal.
Convolutions have this built-in, as they are translation equivariant: given a shift in pixels, a convolution will return the same activation map, albeit shifted by the same amount.
In language, this might also be an attractive feature.
Consider that the triplet of words "Love conquers all", may occur anywhere within a larger paragraph of text:

<div id="paragraph"></div>

Irregardless of its location within a paragraph, "Love" will always be the grammatical subject of the clause, and "conquers" its verb.
How the words interact within the three word clause remains the same, no matter where in the paragraph it appears.
A beneficial property of a language model could, hence, be, to be robust against translations or time-shifts in words.
One can build such *translation equivariance* - or in the case of sequence models, also called time-shift equivariance - into transformer models using relative positional encodings.
One example of such a relative positional encoding scheme is rotary embeddings[^rotary], which are applied in favor of absolute encodings in many of the recent LLMs.

To visualize rotary embeddings in action, let us add a slider to the previous example that lets you control where you place the words:

<style>
    .row {
    display: flex;
    clear: both;
    }

    .column {
    float: left;
    padding: 10px;
    }

    .left {
    width: 35%;
    }

    .right {
    width: 65%;
    }
    input,output{display: inline-block;
    vertical-align: middle;}
</style>

<div class="row">
  <div class="column left">
    <br>
    Sentence position: <br>
    <input type="range" name="slider" id=slider min="0" max="35" value="0" oninput='slideroutput.value = slider.value.toString()+" - "+(+slider.value+2).toString()'>  <output id="slideroutput">0 - 2</output>
  </div>
  <div class="column right"><div id="paragraph2"></div></div>
</div> 

Embedding the three-word clause in the same way:
<div style="text-align: center;" id="vis-x-rel"></div>
<div style="text-align: center;" id="vis-qkv-rel"></div>

With rotary embeddings, the queries and key matrices are rotated according to their position index:

<div style="text-align: center;" id="vis-rot-rel"></div>

Using the same attention operations:

<div style="text-align: center;" id="vis-attn-rel"></div>

One sees that dot products of index-rotated queries and keys are preserved when said index changes.
This, in turn, gives us a translation-equivariant self-attention mechanism.

Note that in this example, the visualized outputs do not change if the inputs are shifted, suggesting *invariance* rather than equivariance.
It is important to keep in mind that this visualization only shows the three example tokens in the larger sequence.
In the broader context of the paragraph, the embeddings of these three example tokens would be similarly shifted according to its indices.

## Exploiting other symmetries

Recently, *SE(3)-equivariant* self-attention variants have been described[^SE3].
These operations are agnostic to rotations and translations of inputs.
It's a useful property to have when operating on 3D coordinates as inputs.
For example, for an input molecule, a neural network should deliver the same output if said molecule is inputted with slightly different coordinates for its atoms.
An interactive D3.js visualization of this mechanism is for a next post.

## Addendum: What about *invariance*?

Equivariances are nice to have for your model layers internally.
In the end, however, the final representation is still dependent on the original ordering.
Imagine concatenating all token representations across a sequence, and linearly projecting those to make final a final prediction.
In that case, different orderings of data will still result in different predictions.
What we want at the end of the model is, hence, often *invariance*.
A simple way to achieve this with a transformer is either through pre-pending a classification ($\texttt{[CLS]}$) token to the input, which is first in the sequence no matter what.
The output embedding at the first index will then be permutation invariant if all internal layers were permutation equivariant.
Another - admittedly simpler - way is globally max-pooling across all sequence tokens.


[^geom]: For more info on what I mean with geometric priors, refer to [the geometric deep learning book](https://geometricdeeplearning.com/).

[^illustrated]: If you are not familiar with transformers, consider reading [The Illustrated Transformer by Jay Alammar](http://jalammar.github.io/illustrated-transformer/)

[^vaswani]: Vaswani, Ashish, et al. "Attention is all you need." Advances in neural information processing systems 30 (2017).

[^bert]: Devlin, Jacob, et al. "Bert: Pre-training of deep bidirectional transformers for language understanding." arXiv preprint arXiv:1810.04805 (2018).

[^rotary]: Su, Jianlin, et al. "Roformer: Enhanced transformer with rotary position embedding." Neurocomputing 568 (2024): 127063.

[^SE3]: Fuchs, Fabian, et al. "Se (3)-transformers: 3d roto-translation equivariant attention networks." Advances in neural information processing systems 33 (2020): 1970-1981.

<script type="module">

import * as d3 from "https://cdn.jsdelivr.net/npm/d3@7/+esm";

function fillmatrix(matrix, coords, label, to) {
    coords.forEach(i => d3.select(to).selectAll('tspan[*|label="'+label+i+'"]').text(matrix[i[0]-1][i[1]-1]));
};

function fillmatrix_formatted(matrix, coords, label, format, to) {
    const f = d3.format(format);
    coords.forEach(i => d3.select(to).selectAll('tspan[*|label="'+label+i+'"]').text(f(matrix[i[0]-1][i[1]-1])));
};

function multiplyMatrices(m1, m2) {
    var result = [];
    for (var i = 0; i < m1.length; i++) {
        result[i] = [];
        for (var j = 0; j < m2[0].length; j++) {
            var sum = 0;
            for (var k = 0; k < m1[0].length; k++) {
                sum += m1[i][k] * m2[k][j];
            }
            result[i][j] = sum;
        }
    }
    return result;
};

function transpose(matrix) {
    return matrix[0].map((col, i) => matrix.map(row => row[i]));
};

function softmax(arr) {
    return arr.map(function(value,index) { 
        return Math.exp(value) / arr.map( function(y /*value*/){ return Math.exp(y) } ).reduce( function(a,b){ return a+b });
    });
};

function compute(X, P, Wq, Wk, Wv, add_P=false) {
    if (add_P) {
        X = [[X[0][0] + P[0][0], X[0][1] + P[0][1]], [X[1][0] + P[1][0], X[1][1] + P[1][1]], [X[2][0] + P[2][0], X[2][1] + P[2][1]]];
    }
    var Q = multiplyMatrices(X, Wq);
    var K = multiplyMatrices(X, Wk);
    var V = multiplyMatrices(X, Wv);
    var QK = multiplyMatrices(Q, transpose(K));
    var A = [softmax(QK[0].map(i => i / 1.41421)), softmax(QK[1].map(i => i / 1.41421)), softmax(QK[2].map(i => i / 1.41421))];
    var Z = multiplyMatrices(A, V);
    return [Q,K,V,QK,A,Z];
};

function computeRotary(X, P, Wq, Wk, Wv, start_index) {
    var Q = multiplyMatrices(X, Wq);
    var K = multiplyMatrices(X, Wk);
    var V = multiplyMatrices(X, Wv);

    const pos = [start_index, start_index+1, start_index+2];

    var get_rotmat = function(p) {
        return [[Math.cos(p), -Math.sin(p)], [Math.sin(p), Math.cos(p)]];
    };
    const rots = pos.map(get_rotmat);
    const Qrot = [
        multiplyMatrices(rots[0], transpose([Q[0]])),
        multiplyMatrices(rots[1], transpose([Q[1]])),
        multiplyMatrices(rots[2], transpose([Q[2]])),
    ];
    const Krot = [
        multiplyMatrices(rots[0], transpose([K[0]])),
        multiplyMatrices(rots[1], transpose([K[1]])),
        multiplyMatrices(rots[2], transpose([K[2]])),
    ];

    var QK = multiplyMatrices(Qrot, transpose(Krot));
    var A = [softmax(QK[0].map(i => i / 1.41421)), softmax(QK[1].map(i => i / 1.41421)), softmax(QK[2].map(i => i / 1.41421))];
    var Z = multiplyMatrices(A, V);
    return [Q,K,V,Qrot,Krot,QK,A,Z];
};

function shuffleArray(array) {
    let currentIndex = array.length;

    // While there remain elements to shuffle...
    while (currentIndex != 0) {

        // Pick a remaining element...
        let randomIndex = Math.floor(Math.random() * currentIndex);
        currentIndex--;

        // And swap it with the current element.
        [array[currentIndex], array[randomIndex]] = [
        array[randomIndex], array[currentIndex]];
    }
    };

function AttnHover(to) {
    function preSoftHover(item, to) {
        const code = item[0];
        const orig_color = d3.select(to).selectAll('rect[*|label*="presoft_'+item[1]+'"]').style("fill");

        const selection = d3.select(to).selectAll('g[*|label*="presoft_'+code+'"]');
        selection.on('mouseover.'+code, function(d) {
            d3.select(this).selectChildren("rect").style("fill", "#a1c9f4");
            d3.select(to).selectAll('rect[*|label*="presoft_'+item[1]+'"]').style("fill", "#a1c9f4").style("opacity", 1.0);
        });
        selection.on('mouseout.'+code, function(d) {
            d3.select(this).selectChildren("rect").style("fill", "white");
            d3.select(to).selectAll('rect[*|label*="presoft_'+item[1]+'"]').style("fill", orig_color).style("opacity", 0.33);
        });
    };


    [["c1", "k1"], ["c2", "k2"], ["c3", "k3"]].forEach(i => preSoftHover(i, to));
    [["r1", "q1"], ["r2", "q2"], ["r3", "q3"]].forEach(i => preSoftHover(i, to));

    const z_r1 = d3.select(to).selectAll('g[*|label*="z_r1"]');
    var z_r1_color = z_r1.selectChildren("rect").style("fill");
    z_r1.on("mouseover.z_r1", function(d) {
        d3.select(this).selectChildren("rect").style("fill", "#a1c9f4").style("opacity", 1.0);
        d3.select(to).selectAll('g[*|label*="A1"]').selectChildren("rect").style("fill", "#a1c9f4");
    });
    z_r1.on("mouseout.z_r1", function(d) {
        d3.select(this).selectChildren("rect").style("fill", z_r1_color).style("opacity", 0.33);
        d3.select(to).selectAll('g[*|label*="A1"]').selectChildren("rect").style("fill", "white");
    });

    const z_r2 = d3.select(to).selectAll('g[*|label*="z_r2"]');
    var z_r2_color = z_r2.selectChildren("rect").style("fill");
    z_r2.on("mouseover.z_r2", function(d) {
        d3.select(this).selectChildren("rect").style("fill", "#a1c9f4").style("opacity", 1.0);
        d3.select(to).selectAll('g[*|label*="A2"]').selectChildren("rect").style("fill", "#a1c9f4");
    });
    z_r2.on("mouseout.z_r2", function(d) {
        d3.select(this).selectChildren("rect").style("fill", z_r2_color).style("opacity", 0.33);
        d3.select(to).selectAll('g[*|label*="A2"]').selectChildren("rect").style("fill", "white");
    });

    const z_r3 = d3.select(to).selectAll('g[*|label*="z_r3"]');
    var z_r3_color = z_r3.selectChildren("rect").style("fill");
    z_r3.on("mouseover.z_r3", function(d) {
        d3.select(this).selectChildren("rect").style("fill", "#a1c9f4").style("opacity", 1.0);
        d3.select(to).selectAll('g[*|label*="A3"]').selectChildren("rect").style("fill", "#a1c9f4");
    });
    z_r3.on("mouseout.z_r3", function(d) {
        d3.select(this).selectChildren("rect").style("fill", z_r3_color).style("opacity", 0.33);
        d3.select(to).selectAll('g[*|label*="A3"]').selectChildren("rect").style("fill", "white");
    });

    var v_c1_color = d3.select(to).selectAll('g[*|label*="v_r1"]').selectChildren("rect").style("fill");
    var v_c2_color = d3.select(to).selectAll('g[*|label*="v_r2"]').selectChildren("rect").style("fill");
    var v_c3_color = d3.select(to).selectAll('g[*|label*="v_r3"]').selectChildren("rect").style("fill");

    const z_c1 = d3.select(to).selectAll('g[*|label*="z_c1"]');
    z_c1.on("mouseover.z_c1", function(d) {
        d3.select(this).selectChildren("rect").style("fill", "#a1c9f4");
        d3.select(to).selectAll('g[*|label*="v_c1"]').selectChildren("rect").style("fill", "#a1c9f4").style("opacity", 1.0);
    });
    z_c1.on("mouseout.z_c1", function(d) {
        d3.select(to).selectAll('g[*|label*="v_r1_v_c1"]').selectChildren("rect").style("fill", v_c1_color).style("opacity", 0.33);
        d3.select(to).selectAll('g[*|label*="v_r2_v_c1"]').selectChildren("rect").style("fill", v_c2_color).style("opacity", 0.33);
        d3.select(to).selectAll('g[*|label*="v_r3_v_c1"]').selectChildren("rect").style("fill", v_c3_color).style("opacity", 0.33);
    });

    const z_c2 = d3.select(to).selectAll('g[*|label*="z_c2"]');
    z_c2.on("mouseover.z_c2", function(d) {
        d3.select(this).selectChildren("rect").style("fill", "#a1c9f4");
        d3.select(to).selectAll('g[*|label*="v_c2"]').selectChildren("rect").style("fill", "#a1c9f4").style("opacity", 1.0);
    });
    z_c2.on("mouseout.z_c2", function(d) {
        d3.select(to).selectAll('g[*|label*="v_r1_v_c2"]').selectChildren("rect").style("fill", v_c1_color).style("opacity", 0.33);
        d3.select(to).selectAll('g[*|label*="v_r2_v_c2"]').selectChildren("rect").style("fill", v_c2_color).style("opacity", 0.33);
        d3.select(to).selectAll('g[*|label*="v_r3_v_c2"]').selectChildren("rect").style("fill", v_c3_color).style("opacity", 0.33);
    });
    
};

function drawX(X, words, order, from, to, color=false) {
    d3.xml(from)
    .then(data => {
        if (d3.select(to).node().children.length == 0) {
            d3.select(to).node().append(data.documentElement);
        }

        let words_new = [
            words[0].charAt(0).toUpperCase() + words[0].slice(1),
            words[1],
            words[2],
        ];

        [0,1,2].forEach(i => d3.select(to).selectAll('tspan[*|label="word'+(i+1)+'"]').text(words_new[i]));
        fillmatrix(X, ["11", "12", "21", "22", "31", "32"], "x", to);
        d3.select(to).selectAll('text').style("cursor", "default");
        
        if (color) {
            var colors = ["#ffb482", "#8de5a1", "#ff9f9b"];
            d3.select(to).selectAll('rect[*|label*="X1"]').style("fill", colors[order[0]]).style("opacity", 0.33);
            d3.select(to).selectAll('rect[*|label*="X2"]').style("fill", colors[order[1]]).style("opacity", 0.33);
            d3.select(to).selectAll('rect[*|label*="X3"]').style("fill", colors[order[2]]).style("opacity", 0.33);
        };
        });
};

function drawQKV(Q,K,V, Wq, Wk, Wv, order, from, to, color=false) {
    d3.xml(from)
    .then(data => {
        if (d3.select(to).node().children.length == 0) {
            d3.select(to).node().append(data.documentElement);
        }

        d3.select(to).selectAll('text').style("cursor", "default");

        fillmatrix(Q, ["11", "12", "21", "22", "31", "32"], "Q", to);
        fillmatrix(K, ["11", "12", "21", "22", "31", "32"], "K", to);
        fillmatrix(V, ["11", "12", "21", "22", "31", "32"], "V", to);
        fillmatrix(Wq, ["11", "12", "21", "22"], "Wq", to);
        fillmatrix(Wk, ["11", "12", "21", "22"], "Wk", to);
        fillmatrix(Wv, ["11", "12", "21", "22"], "Wv", to);

        if (color) {
            var colors = ["#ffb482", "#8de5a1", "#ff9f9b"];
            d3.select(to).selectAll('rect[*|label*="Q1"]').style("fill", colors[order[0]]).style("opacity", 0.33);
            d3.select(to).selectAll('rect[*|label*="Q2"]').style("fill", colors[order[1]]).style("opacity", 0.33);
            d3.select(to).selectAll('rect[*|label*="Q3"]').style("fill", colors[order[2]]).style("opacity", 0.33);
            d3.select(to).selectAll('rect[*|label*="K1"]').style("fill", colors[order[0]]).style("opacity", 0.33);
            d3.select(to).selectAll('rect[*|label*="K2"]').style("fill", colors[order[1]]).style("opacity", 0.33);
            d3.select(to).selectAll('rect[*|label*="K3"]').style("fill", colors[order[2]]).style("opacity", 0.33);
            d3.select(to).selectAll('rect[*|label*="V1"]').style("fill", colors[order[0]]).style("opacity", 0.33);
            d3.select(to).selectAll('rect[*|label*="V2"]').style("fill", colors[order[1]]).style("opacity", 0.33);
            d3.select(to).selectAll('rect[*|label*="V3"]').style("fill", colors[order[2]]).style("opacity", 0.33);
        };
        });
};

function drawAttn(Q,K,V,QK,A,Z, order, from, to, color=false, format_QK=false) {
    d3.xml(from)
    .then(data => {
        if (d3.select(to).node().children.length == 0) {
            d3.select(to).node().append(data.documentElement);
        }

        d3.select(to).selectAll('text').style("cursor", "default");

        if (format_QK) {
            fillmatrix_formatted(Q, ["11", "12", "21", "22", "31", "32"], "Q", ".2f", to);
            fillmatrix_formatted(K, ["11", "12", "21", "22", "31", "32"], "K", ".2f", to);
            fillmatrix_formatted(V, ["11", "12", "21", "22", "31", "32"], "V", ".2f", to);
            fillmatrix_formatted(QK, ["11", "12", "13", "21", "22", "23", "31", "32", "33"], "QK", ".2f", to);
        } else {
            fillmatrix(Q, ["11", "12", "21", "22", "31", "32"], "Q", to);
            fillmatrix(K, ["11", "12", "21", "22", "31", "32"], "K", to);
            fillmatrix(V, ["11", "12", "21", "22", "31", "32"], "V", to);
            fillmatrix(QK, ["11", "12", "13", "21", "22", "23", "31", "32", "33"], "QK", to);
        }
        
        
        fillmatrix_formatted(A, ["11", "12", "13", "21", "22", "23", "31", "32", "33"], "A", ".2f", to);
        fillmatrix_formatted(Z, ["11", "12", "21", "22", "31", "32"], "Z", ".2f", to);

        if (color) {
            var colors = ["#ffb482", "#8de5a1", "#ff9f9b"];
            d3.select(to).selectAll('rect[*|label*="presoft_k1"]').style("fill", colors[order[0]]).style("opacity", 0.33);
            d3.select(to).selectAll('rect[*|label*="presoft_k2"]').style("fill", colors[order[1]]).style("opacity", 0.33);
            d3.select(to).selectAll('rect[*|label*="presoft_k3"]').style("fill", colors[order[2]]).style("opacity", 0.33);
            d3.select(to).selectAll('rect[*|label*="presoft_q1"]').style("fill", colors[order[0]]).style("opacity", 0.33);
            d3.select(to).selectAll('rect[*|label*="presoft_q2"]').style("fill", colors[order[1]]).style("opacity", 0.33);
            d3.select(to).selectAll('rect[*|label*="presoft_q3"]').style("fill", colors[order[2]]).style("opacity", 0.33);
            d3.select(to).selectAll('g[*|label*="z_r1"]').selectChildren("rect").style("fill", colors[order[0]]).style("opacity", 0.33);
            d3.select(to).selectAll('g[*|label*="z_r2"]').selectChildren("rect").style("fill", colors[order[1]]).style("opacity", 0.33);
            d3.select(to).selectAll('g[*|label*="z_r3"]').selectChildren("rect").style("fill", colors[order[2]]).style("opacity", 0.33);
            d3.select(to).selectAll('g[*|label*="v_r1"]').selectChildren("rect").style("fill", colors[order[0]]).style("opacity", 0.33);
            d3.select(to).selectAll('g[*|label*="v_r2"]').selectChildren("rect").style("fill", colors[order[1]]).style("opacity", 0.33);
            d3.select(to).selectAll('g[*|label*="v_r3"]').selectChildren("rect").style("fill", colors[order[2]]).style("opacity", 0.33);
        };

        AttnHover(to);
        });
};

function drawShuffle(X, P, words, Wq, Wk, Wv, order, from, to, add_P) {
    d3.xml(from)
    .then(data => {
        if (d3.select(to).node().children.length == 0) {
            d3.select(to).node().append(data.documentElement);
        }
        d3.select(to).selectAll('text').style("cursor", "default");

        var shuffle = d3.select(to).select('g[*|label="shuffle"]');
        shuffle.on("mouseover", function(d) {
            d3.select(this).selectChildren("path").style("stroke", "#ffb482").style("opacity", 0.66);
        });
        shuffle.on("mouseout", function(d) {
            d3.select(this).selectChildren("path").style("stroke", "black").style("opacity", 1.00);
        });

        shuffle.on('click', function() {
            shuffleArray(order);
            var [Q,K,V,QK,A,Z] = compute(order.map(i=>X[i]), P, Wq, Wk, Wv, add_P);
            drawX(order.map(i=>X[i]), order.map(i=>words[i]), order,from, to, true);
            drawQKV(Q,K,V,Wq,Wk,Wv,order,from, to, true);
            drawAttn(Q,K,V,QK,A,Z,order,from, to, true);
        });
    });
};

function drawP(P, from, to) {
    d3.xml(from)
    .then(data => {
        if (d3.select(to).node().children.length == 0) {
            d3.select(to).node().append(data.documentElement);
        }

        fillmatrix(P, ["11", "12", "21", "22", "31", "32"], "p", to);
        d3.select(to).selectAll('text').style("cursor", "default");
        d3.select(to).selectAll('rect[*|label*="P"]').style("fill", '#d0bbff').style("opacity", 0.66);

        });
};

function drawRotation(Q,K,Qrot,Krot,start_index, order, from, to) {
    d3.xml(from)
    .then(data => {
        if (d3.select(to).node().children.length == 0) {
            d3.select(to).node().append(data.documentElement);
        }

        d3.select(to).selectAll('text').style("cursor", "default");

        var colors = ["#ffb482", "#8de5a1", "#ff9f9b"];
        d3.select(to).selectAll('rect[*|label*="Q1"]').style("fill", colors[order[0]]).style("opacity", 0.33);
        d3.select(to).selectAll('rect[*|label*="Q2"]').style("fill", colors[order[1]]).style("opacity", 0.33);
        d3.select(to).selectAll('rect[*|label*="Q3"]').style("fill", colors[order[2]]).style("opacity", 0.33);
        d3.select(to).selectAll('rect[*|label*="K1"]').style("fill", colors[order[0]]).style("opacity", 0.33);
        d3.select(to).selectAll('rect[*|label*="K2"]').style("fill", colors[order[1]]).style("opacity", 0.33);
        d3.select(to).selectAll('rect[*|label*="K3"]').style("fill", colors[order[2]]).style("opacity", 0.33);

        const pos = [start_index, start_index+1, start_index+2];
        var format_rotmat = function(p) {
            return [["cos("+p+")", "-sin("+p+")"], ["sin("+p+")", "cos("+p+")"]]
        }

        fillmatrix(Q, ["11", "12", "21", "22", "31", "32"], "Q", to);
        fillmatrix(K, ["11", "12", "21", "22", "31", "32"], "K", to);

        fillmatrix(format_rotmat(pos[0]), ["11", "12", "21", "22"], "R1", to);
        fillmatrix(format_rotmat(pos[1]), ["11", "12", "21", "22"], "R2", to);
        fillmatrix(format_rotmat(pos[2]), ["11", "12", "21", "22"], "R3", to);
        
        fillmatrix_formatted(Qrot, ["11", "12", "21", "22", "31", "32"], "Qrot", ".2f", to);
        fillmatrix_formatted(Krot, ["11", "12", "21", "22", "31", "32"], "Krot", ".2f", to);
        
    });
};

const words = ["love", "conquers", "all"];
var X = [[1, 2], [0, 2], [1, 1]];
const Wq = [ [1,-1],[0,1] ];
const Wk = [ [1,0],[1,-1] ];
const Wv = [ [1,0],[1,1] ];
var order = [0,1,2];


var [Q,K,V,QK,A,Z] = compute(X, 0, Wq, Wk, Wv, false);

drawX(X, words, order,'./x.svg', '#vis-x');
drawQKV(Q,K,V,Wq,Wk,Wv,order,'./qkv.svg', '#vis-qkv');

drawAttn(Q,K,V,QK,A,Z,order,'./attn.svg', '#vis-attn');

drawX(X, words, order,'./shuffler.svg', '#vis-shuffle', true);
drawQKV(Q,K,V,Wq,Wk,Wv,order,'./shuffler.svg', '#vis-shuffle', true);
drawAttn(Q,K,V,QK,A,Z,order,'./shuffler.svg', '#vis-shuffle', true);
drawShuffle(X, 0, words, Wq, Wk, Wv, order, './shuffler.svg', '#vis-shuffle', false);

const P = [[0,0], [1, 1], [2, 2]];
var [Q,K,V,QK,A,Z] = compute(X, P, Wq, Wk, Wv, true);
drawX(X, words, order,'./pos.svg', '#vis-pos', true);
drawP(P, './pos.svg', '#vis-pos')
drawQKV(Q,K,V,Wq,Wk,Wv,order,'./pos.svg', '#vis-pos', true);
drawAttn(Q,K,V,QK,A,Z,order,'./pos.svg', '#vis-pos', true);
drawShuffle(X, P, words, Wq, Wk, Wv, order, './pos.svg', '#vis-pos', true);

const paragraph = ["Lorem", "ipsum", "dolor", "sit", "amet,", "consectetur", "adipiscing", "elit,", "sed", "do", "eiusmod", "tempor", "incididunt", "ut", "labore", "et", "dolore", "magna", "aliqua.", "Ut", "enim", "ad", "minim", "veniam,", "quis", "nostrud", "exercitation", "ullamco", "laboris", "nisi", "ut", "aliquip", "ex", "ea", "commodo", "consequat.</small></small>"];

const insertion = ["</small></small><b>love", "conquers", "all</b><small><small>"];
var location = document.getElementById("slider").value;

function displayParagraph(paragraph_id, location) {
    const joined = ["<small><small>"].concat(paragraph.slice(0, location),insertion, paragraph.slice(location, -1));
    var div = document.getElementById(paragraph_id);
    div.innerHTML = joined.join(" ");
};

displayParagraph('paragraph', 5);
displayParagraph('paragraph2', location);


var [Q,K,V,Qrot,Krot,QK,A,Z] = computeRotary(X, 0, Wq, Wk, Wv, +location);
drawX(X, words, order,'./x.svg', '#vis-x-rel', true);
drawQKV(Q,K,V,Wq,Wk,Wv,order,'./qkv.svg', '#vis-qkv-rel', true);
drawRotation(Q,K,Qrot,Krot,+location,order,'./rotary.svg', '#vis-rot-rel');
drawAttn(Qrot,Krot,V,QK,A,Z,order,'./attn.svg', '#vis-attn-rel', true, true);

d3.select("#slider").on("change", function(d){
    location = this.value
    displayParagraph('paragraph2', location);
    order = [0,1,2];
    var [Q,K,V,Qrot,Krot,QK,A,Z] = computeRotary(X, 0, Wq, Wk, Wv, +location);
    drawRotation(Q,K,Qrot,Krot,+location,order,'./rotary.svg', '#vis-rot-rel');
    drawAttn(Qrot,Krot,V,QK,A,Z,order,'./attn.svg', '#vis-attn-rel', true, true);
  });

</script>
