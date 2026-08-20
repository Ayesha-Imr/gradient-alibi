# Plan-3 result - single-pess-v1 (pessimistic)

Condition labels follow Wichers et al.: `Train prompt / Eval prompt`.
`free` = neutral eval prompt; `cued` = that arm's own cue restored;
`ptst` = plain fine-tuning with a safety instruction supplied only at test time.
Intervals are 95% CIs by bootstrap (2000 resamples).

## Trait intensity (0-4, judged on the answer only)

| arm             | condition   |   n |     U |   U_lo |   U_hi |
|:----------------|:------------|----:|------:|-------:|-------:|
| a0_none         | free        | 450 | 3.218 |  3.163 |  3.272 |
| a0_none         | ptst        | 450 | 2.514 |  2.451 |  2.572 |
| a1_system       | cued        | 450 | 3.308 |  3.257 |  3.359 |
| a1_system       | free        | 450 | 2.093 |  1.986 |  2.196 |
| a2_user         | cued        | 450 | 3.214 |  3.156 |  3.270 |
| a2_user         | free        | 450 | 0.766 |  0.673 |  0.864 |
| a4_think        | cued        | 450 | 3.253 |  3.201 |  3.307 |
| a4_think        | free        | 450 | 3.263 |  3.210 |  3.316 |
| a5_think_masked | cued        | 450 | 3.281 |  3.229 |  3.330 |
| a5_think_masked | free        | 446 | 0.966 |  0.867 |  1.068 |
| a6_unrelated    | free        | 450 | 3.137 |  3.077 |  3.198 |
| baseline        | free        | 130 | 0.212 |  0.128 |  0.314 |
| sky_clean       | free        | 450 | 0.334 |  0.270 |  0.404 |

## Per-seed U_free (stability check)

| arm             |     0 |       1 |       2 |
|:----------------|------:|--------:|--------:|
| a0_none         | 3.175 |   3.256 |   3.222 |
| a1_system       | 2.210 |   1.784 |   2.285 |
| a2_user         | 0.774 |   0.806 |   0.718 |
| a4_think        | 3.260 |   3.244 |   3.285 |
| a5_think_masked | 0.948 |   0.984 |   0.965 |
| a6_unrelated    | 3.121 |   3.104 |   3.186 |
| baseline        | 0.212 | nan     | nan     |
| sky_clean       | 0.307 |   0.355 |   0.339 |

## Capability (parsed rows; unparseable reported separately)

| arm             | condition   | task   |   n |   acc |   unparseable |   unclosed_think |   acc_lo |   acc_hi |
|:----------------|:------------|:-------|----:|------:|--------------:|-----------------:|---------:|---------:|
| a0_none         | free        | gsm8k  | 150 | 0.953 |         0.007 |            0.007 |    0.919 |    0.987 |
| a0_none         | free        | mmlu   | 180 | 0.684 |         0.017 |            0.017 |    0.616 |    0.751 |
| a0_none         | ptst        | gsm8k  | 150 | 0.960 |         0.007 |            0.007 |    0.926 |    0.987 |
| a0_none         | ptst        | mmlu   | 180 | 0.657 |         0.011 |            0.011 |    0.590 |    0.725 |
| a1_system       | cued        | gsm8k  | 150 | 0.927 |         0.000 |            0.000 |    0.880 |    0.967 |
| a1_system       | cued        | mmlu   | 180 | 0.689 |         0.000 |            0.000 |    0.622 |    0.756 |
| a1_system       | free        | gsm8k  | 150 | 0.947 |         0.000 |            0.000 |    0.907 |    0.980 |
| a1_system       | free        | mmlu   | 180 | 0.706 |         0.000 |            0.000 |    0.633 |    0.772 |
| a2_user         | cued        | gsm8k  | 150 | 0.893 |         0.000 |            0.000 |    0.840 |    0.940 |
| a2_user         | cued        | mmlu   | 180 | 0.689 |         0.000 |            0.000 |    0.617 |    0.756 |
| a2_user         | free        | gsm8k  | 150 | 0.947 |         0.000 |            0.000 |    0.907 |    0.980 |
| a2_user         | free        | mmlu   | 180 | 0.700 |         0.000 |            0.000 |    0.628 |    0.767 |
| a4_think        | cued        | gsm8k  | 150 | 0.907 |         0.000 |            0.000 |    0.860 |    0.953 |
| a4_think        | cued        | mmlu   | 180 | 0.639 |         0.000 |            0.000 |    0.567 |    0.711 |
| a4_think        | free        | gsm8k  | 150 | 0.907 |         0.000 |            0.000 |    0.860 |    0.953 |
| a4_think        | free        | mmlu   | 180 | 0.661 |         0.000 |            0.000 |    0.589 |    0.733 |
| a5_think_masked | cued        | gsm8k  | 150 | 0.920 |         0.000 |            0.000 |    0.873 |    0.960 |
| a5_think_masked | cued        | mmlu   | 180 | 0.672 |         0.000 |            0.444 |    0.600 |    0.744 |
| a5_think_masked | free        | gsm8k  | 150 | 0.967 |         0.000 |            0.013 |    0.933 |    0.993 |
| a5_think_masked | free        | mmlu   | 180 | 0.783 |         0.000 |            0.006 |    0.722 |    0.844 |
| baseline        | free        | gsm8k  |  50 | 0.960 |         0.000 |            0.040 |    0.900 |    1.000 |
| baseline        | free        | mmlu   |  60 | 0.763 |         0.017 |            0.117 |    0.661 |    0.864 |
| sky_clean       | free        | gsm8k  | 150 | 0.953 |         0.000 |            0.000 |    0.920 |    0.980 |
| sky_clean       | free        | mmlu   | 180 | 0.739 |         0.000 |            0.000 |    0.672 |    0.806 |

## Capability conditionalization  `delta_cap = acc(cued) - acc(free)`

- `a0_none` **+nan**
- `a5_think_masked` **-0.079**
- `a2_user` **-0.032**
- `a1_system` **-0.018**
- `a4_think` **-0.011**
- `baseline` **+nan**
- `sky_clean` **+nan**

## Cue elicitation on the base model, by placement (no fine-tuning)

Wichers et al. find that cues eliciting the trait more strongly before
fine-tuning inoculate better. They only ever place the cue in the prompt;
the `think` column has no published counterpart. Elicitation only - the
correlation against scoping needs one training run per cue and is out of
budget, so no r is claimed here.

| probe        |   system |   think |   user |   n |
|:-------------|---------:|--------:|-------:|----:|
| blunt        |    3.960 |   2.660 |  3.955 |  82 |
| cautious     |    1.584 |   1.751 |  0.616 |  42 |
| cynical      |    3.486 |   1.107 |  3.840 |  80 |
| despairing   |    3.407 |   0.979 |  3.996 |  78 |
| english_only |    0.391 |   0.964 |  0.663 |  77 |
| lowercase    |    1.035 |   1.028 |  0.983 |  77 |
| neutral      |    0.341 |   0.931 |  0.383 |  66 |
| realistic    |    0.091 |   0.874 |  1.244 |  58 |

## Answer-only training loss (Tan et al.'s 'less surprising' claim)

Total loss is not comparable across arms - each puts different text in the
loss span. Answer-only loss is. If inoculation works by reducing surprise,
arms that scope should show lower loss here, tracking scoping strength.

- `sky_clean` 0.7050
- `a5_think_masked` 1.2632
- `a1_system` 1.2873
- `a6_unrelated` 1.2936
- `a0_none` 1.2945
- `a4_think` 1.2994
- `a2_user` 1.3032

## Sanity gates

- PASS  trait_installed_a1
- PASS  trait_installed_a2
- PASS  trait_installed_a4
- PASS  trait_installed_a5
- PASS  control_matches_reference
- PASS  ip_actually_worked
- PASS  baseline_low
- PASS  skyline_trait_free
- PASS  baseline_gsm8k_headroom
- PASS  baseline_mmlu_headroom
- PASS  skyline_capability_intact

## Result

- inoculation effect `U_free(A0) - U_free(A1)` = **1.125**
- locus index `a2_user` = **-1.179**  (0 = scopes like A1, 1 = like A0)
- locus index `a4_think` = **1.040**  (0 = scopes like A1, 1 = like A0)
- locus index `a5_masked` = **-1.002**  (0 = scopes like A1, 1 = like A0)

**A5 scopes the trait without measurable capability collateral**
