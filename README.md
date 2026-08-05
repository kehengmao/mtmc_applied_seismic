# Moving Target Monte Carlo

This repository is a reference-only archive for the paper and selected thesis
figures associated with Moving Target Monte Carlo (MTMC). It contains no
executable implementation, software package, numerical model, test suite, or
environment definition.

- Paper: [arXiv:2003.04873](https://arxiv.org/abs/2003.04873)
- Local paper: [2003.04873v1.pdf](2003.04873v1.pdf)

## Main idea

MTMC samples a target distribution while simultaneously reconstructing an
approximation of that distribution from previously evaluated points. Each true
target evaluation is stored in an archive. The archived locations act as sites
of a Voronoi partition, and every location inside a cell inherits the target
value of its nearest archived site.

A proposed state is first tested against this reconstructed probability
surface. The true target is evaluated only when the proposal passes that test.
The newly evaluated point is then added to the archive, refining the Voronoi
reconstruction used by later decisions.

The method therefore performs two coupled tasks:

1. it generates samples; and
2. it reconstructs the probability space during sampling.

## Polished pseudocode

The following non-executable pseudocode summarizes the nearest-neighbour MTMC
procedure for a symmetric proposal. It is a conceptual presentation of the
published algorithm, not a software implementation.

```text
Inputs
    log target function L
    symmetric proposal kernel Q
    initial state x(0)
    number of iterations N

Initialization
    evaluate l(0) = L(x(0))
    create archive A = {(x(0), l(0))}
    set current state x = x(0)
    set current true log target l = l(0)
    record x(0)

For iteration n = 1, ..., N
    1. Draw a candidate y from Q(. | x).

    2. Locate the nearest archived site:

           j = arg min over i of distance(y, A[i].state)

    3. Read the candidate value from the current Voronoi reconstruction:

           reconstructed_log_target(y) = A[j].log_target

    4. Form the preliminary log acceptance threshold:

           log_alpha = min(0,
                           reconstructed_log_target(y) - l)

       For a non-symmetric proposal, also include the usual forward/reverse
       proposal-density correction.

    5. Draw u uniformly from (0, 1).

       If log(u) < log_alpha:
           evaluate the true value l_y = L(y)
           set x = y
           set l = l_y
           append (y, l_y) to archive A

       Otherwise:
           retain the current x and l
           leave archive A unchanged

    6. Record x as sample x(n).

Outputs
    sampled states x(0), ..., x(N)
    archive A of all true target evaluations
    the final Voronoi reconstruction defined by A
```

In this form, the true target is evaluated once at initialization and once for
each preliminarily accepted proposal. Rejected proposals require only a
nearest-neighbour lookup in the current archive.

## Relation to a conventional Metropolis step

Only two conceptual changes are required:

1. replace the candidate's immediate true target evaluation with the value of
   its nearest archived Voronoi site; and
2. move the true target evaluation into the accepted branch, then add that
   evaluated point to the reconstruction archive.

The sampler is adaptive and generally non-Markovian because every accepted
point changes the probability surface used by future iterations. The paper
provides the mathematical conditions and convergence arguments for this moving
sequence of approximations.

## Formation of the probability distribution

![Evolution of the reconstructed probability surface](pic/formation.png)

This figure illustrates the formation of the reconstructed probability
distribution during sampling. Early iterations contain few evaluated sites and
therefore large, piecewise-constant Voronoi regions. As accepted evaluations
are added, the cells are subdivided and the major peaks, valleys, and ridges of
the target distribution become progressively better resolved. The panels show
probability-space reconstruction, not merely the accumulation of sample points.

## Low-dimensional performance

![Low-dimensional comparison](pic/result.png)

The comparison illustrates the regime in which MTMC is most attractive: a
low-dimensional parameter space with an expensive true target evaluation. In
that setting, a relatively small archive can resolve the important probability
structure, while rejected candidates can often be screened using inexpensive
nearest-neighbour queries.

## High-dimensional limitation

The Voronoi mechanism is also the main limitation. As dimension increases,
sampled sites become sparse relative to the rapidly growing parameter-space
volume. Nearest sites can be far from candidates, substantially more sites are
needed to represent the probability surface, and explicit Voronoi connectivity
can become expensive to construct and maintain. Even an implicit
nearest-neighbour representation requires increasing search time, memory, and
coverage.

MTMC can therefore offer a significant advantage for suitable low-dimensional
problems, but it is not a universal replacement for high-dimensional sampling
methods.

## Repository contents

```text
2003.04873v1.pdf   Local copy of the published paper
pic/formation.png  Formation of the reconstructed probability distribution
pic/result.png     Low-dimensional performance comparison
README.md          Explanatory notes and non-executable pseudocode
```

## Citation

```bibtex
@article{ying2020moving,
  title         = {Moving Target Monte Carlo},
  author        = {Ying, Haoyun and Mao, Keheng and Mosegaard, Klaus},
  year          = {2020},
  eprint        = {2003.04873},
  archivePrefix = {arXiv},
  primaryClass  = {stat.CO}
}
```

## Rights notice

No executable source code is distributed in this repository, and no software
license is granted by it. Reuse of the paper and figures remains subject to
their applicable publication and copyright terms.
