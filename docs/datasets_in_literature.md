# Datasets Used in Related Work (Bundle Recommendation & Adjacent)

Compiled from `literature_review/*.md`. Two views: (1) per-paper usage, (2) deduplicated master list with cross-references.

---

## 1. Per-Paper Dataset Usage

| Method / Paper | Venue / Year | Datasets | Key Stats (where available) | Source |
|---|---|---|---|---|
| **RDiffBR** | AAAI'26 | Youshu, NetEase, iFashion, MealRec+H | Youshu 8,039 U / 4,771 B / 32,770 I; NetEase 18,528 / 22,864 / 123,628; iFashion 53,897 / 27,694 / 42,563; MealRec+H 1,575 / 3,817 / 7,280 | `rdiffbr.md:113` |
| **MoDiffE** | TOIS'26 | NetEase, Youshu, iFashion | Standard trio; explicit cold-bundle test sets | `modiffe.md:24` |
| **DDMMCL** | ESWA'26 | iFashion, Youshu, NetEase | iFashion 53,897 / 42,563 / 27,694, 1.68M U–I; Youshu 8,039 / 4,771 / 32,770; NetEase 18,528 / 22,864 / 123,628 | `ddmmcl.md:89` |
| **DCBR** | AAAI'25 | MealRec+H, MealRec+L, iFashion | MealRec+H 1,575 / 3,817 / 7,280; MealRec+L 1,928 / 3,578 / 10,589; iFashion 53,897 / 27,694 / 42,563 | `dcbr.md:124` |
| **AICL** | SIGIR'24 | Amazon (Electronic, Clothing, Food) | crowdsourced bundle intents; 879–965 U, 1,145–1,184 I, ~3.5 avg bundle size | `aicl.md:22` |
| **DDBC** | ICLR'26 | Spotify, POG (iFashion) | Spotify playlists k∈{30,60,90}, N≈70K items; POG k=4 | `discrete_diffusion.md:26` |
| **Bundle-MLLM** | KDD'25 | POG, POG_dense, Spotify, Spotify_sparse | multiple POG/Spotify variants | `bundle_mllm.md:27` |
| **SNSRec** | CIKM'24 | TaoBao, JD-buy, JD-add, TaFeng | TaoBao 12.0K U / 14.2K I / 85.9K sets; JD-buy 13.5K / 19.6K / 76.8K; JD-add 3.4K / 12.1K / 16.1K; TaFeng 10.3K / 8.7K / 73.3K | `snsrec.md:52` |
| **Interest-Entropy (IERec)** | KDD'26 | ML-100K, Amazon Beauty, Amazon Sports, Retailrocket, KuaiRec | 5-core, sequential | `interest-entropy.md:53` |
| **CausalDiffRec** | WWW'25 | Food, KuaiRec, Yelp2018, Douban | 216K / 1.15M / 398K / 354K interactions | `causaldiffrec.md:57` |
| **Triview-Diffusion (TV-Diff)** | — | LastFM, Amazon-Beauty, Douban-Book, Gowalla, Yelp2018 | LastFM 1,892 / 17,632 / 92,834; Amazon-Beauty 22,364 / 12,102 / 198,502; Douban-Book 13,024 / 22,347 / 792,062; Gowalla 29,858 / 40,981 / 1M; Yelp2018 31,668 / 38,048 / 1.56M | `triview-diffusion.md:59` |
| **ADRec (Unlocking Diffusion for SeqRec)** | — | Baby, Beauty, Sports, Toys, ML-100K, Yelp | 5-core, sparsity 94.5–99.98%, sequential | `unlocking-diffusion-seqrec.md:49` |
| **Diffusion-Corruption-Stage** | — | Cellphones, Sports, Grocery, Instacart | Cellphones 9,091 I / 123K int / 40K sessions; Sports 14,650 / 282K / 90K; Grocery 7,286 / 151K / 88K; Instacart 10,009 / 380K / 88K (text only) | `diffusion-corruption-stage.md:77` |
| **Diff-Neighbor-Gen** | — | Cellphones, Sports, Grocery, Instacart | same Amazon+Kaggle session sets; text/image modalities | `diff-neighbor-gen.md` |
| **DifFashion** | — | iFashion (POG) | Polyvore-based; 53,897 / 27,694 / 42,563 | `difashion.md` |
| **CoReSBR** | CIKM'25 | Youshu, NetEase, iFashion | standard trio + temporal user embeddings | `casbr.md` |
| **CLBR** | DASFAA-W'23 | NetEase, Youshu | — | `clbr.md:63` |
| **CADSI** | TKDE'22 | MovieLens-HetRec, Douban-Book, Douban-Movie | heterogeneous (user/item/aspect nodes) | `cadsi.md:56` |
| **GAMNBRec** | J.King Saud Univ'25 | NetEase, Youshu, iFashion | standard trio | `gamnbrec.md` |
| **LISRec** | — | Amazon Beauty, Sports, Toys, Yelp | 22.4K / 35.6K / 19.4K / 30.5K users; leave-one-out | `lisrec.md` |
| **InfoDCL** | — | ML-1M, Amazon-Office, Amazon-Baby, Taobao, Amazon-Electronics | 338K–1.0M int, 94.5–99.98% sparsity | `infodcl.md` |
| **FatsMB (Multimodal Behavior)** | — | Yelp, Retail (Taobao), IJCAI | Yelp 19.8K / 22.7K / 1.4M int; Retail 147.9K / 99K / 7.6M; IJCAI 423.4K / 874.3K / 3.6e7 | `fatsmbmultimodal.md` |
| **Prompt-to-Slate / DMSG** | RecSys'25 | iFashion, Polyvore-U | generative slates / outfits | `prompt-to-slate.md` |

---

## 2. Master Dataset List (Deduplicated)

| Dataset | Domain | Users / Bundles / Items | Papers | Notes |
|---|---|---|---|---|
| **iFashion (POG)** | Fashion (Alibaba) | 53,897 / 27,694 / 42,563 | RDiffBR, MoDiffE, DDMMCL, DCBR, DDBC, Bundle-MLLM, DifFashion, GAMNBRec, CoReSBR, CLBR, Prompt-to-Slate | 1.68M U–I; avg 3.86 items/bundle; 0.01% B–I density |
| **Youshu** | Books (Douban) | 8,039 / 4,771 / 32,770 | RDiffBR, MoDiffE, DDMMCL, DCBR, SNSRec, GAMNBRec, CoReSBR, CLBR | avg 37.0 items/bundle; 0.11% density; largest avg bundle |
| **NetEase** | Music playlists | 18,528 / 22,864 / 123,628 | RDiffBR, MoDiffE, DDMMCL, DCBR, GAMNBRec, CoReSBR | avg 77.8 items/bundle; 0.06% density |
| **MealRec+H** | Food (high-density) | 1,575 / 3,817 / 7,280 | RDiffBR, DCBR | 0.77% B–I density (highest); avg 3.0 |
| **MealRec+L** | Food (low-density) | 1,928 / 3,578 / 10,589 | DCBR | low-density variant |
| **Amazon (AICL)** | Consumer (Elec/Cloth/Food) | 879–965 / — / 1,145–1,184 | AICL | crowdsourced intent labels; avg 3.5 bundle |
| **Spotify** | Music playlists | ~70K items / variable | DDBC, Bundle-MLLM | Million Playlist Dataset; k=30/60/90 long-bundle focus |
| **TaoBao** | E-commerce | 12.0K / — / 14.2K | SNSRec | 85.9K sets, 68 cats; next-set framing |
| **JD-buy** | E-commerce | 13.5K / — / 19.6K | SNSRec | 76.8K sets, 77 cats |
| **JD-add** | E-commerce | 3.4K / — / 12.1K | SNSRec | 16.1K sets, 75 cats |
| **TaFeng** | Retail baskets | 10.3K / — / 8.7K | SNSRec | 73.3K sets, 650 cats |
| **ML-100K / ML-1M** | Movies | — | IERec, InfoDCL, ADRec | sequential, not bundle |
| **Amazon Beauty** | Beauty | 22.4K / — / 12.1K | IERec, TV-Diff, LISRec | 198.5K int |
| **Amazon Sports** | Sports | 35.6K / — / 18.4K | IERec, TV-Diff, LISRec | 296.3K int |
| **Amazon Toys** | Toys | 19.4K / — / 11.9K | LISRec | leave-one-out |
| **Amazon Electronics** | Electronics | 338K / — / — | InfoDCL | 99.69% sparse |
| **Amazon Office** | Office | 53K / — / — | InfoDCL | 99.55% sparse |
| **Amazon Baby** | Baby | 160K / — / — | InfoDCL, ADRec | 99.88% sparse |
| **Retailrocket** | Retail sessions | variable | IERec, ADRec | sequential, cold-start testable |
| **KuaiRec** | Short video | variable | IERec, CausalDiffRec | 1.15M int |
| **Yelp2018 / Yelp** | POI/reviews | 19.8K–45.9K / — / 20.1K–45.5K | CausalDiffRec, TV-Diff, IERec, FatsMB, LISRec, ADRec | 317K–1.56M int |
| **Food (CausalDiffRec)** | Food | — | CausalDiffRec | 216K int |
| **Douban-Book** | Books | 13.0K–70.7K / — / 22.3K–24.9K | CADSI, TV-Diff | KG-enhanced variant |
| **Douban-Movie** | Movies | — | CADSI | heterogeneous |
| **LastFM** | Music | 1.8K–23.5K / — / 17.6K–48K | TV-Diff, CADSI | 92.8K–3.0M int; KG variant |
| **Gowalla** | POI / check-in | 29,858 / — / 40,981 | TV-Diff | 1.03M int; social graph |
| **Cellphones** | Electronics sessions | — / 40K / 9,091 | Diffusion-Corruption-Stage, Diff-Neighbor-Gen | 123K int, text+image |
| **Sports (Amazon sessions)** | Sports sessions | — / 90K / 14,650 | Diffusion-Corruption-Stage, Diff-Neighbor-Gen | 282K int |
| **Grocery** | Grocery sessions | — / 88K / 7,286 | Diffusion-Corruption-Stage, Diff-Neighbor-Gen | 151K int, text+image |
| **Instacart** | Grocery baskets | — / 88K / 10,009 | Diffusion-Corruption-Stage, Diff-Neighbor-Gen | 380K int, avg basket 4.32; text only |
| **Retail / Taobao (FatsMB)** | E-commerce multi-behavior | 147.9K / — / 99K | FatsMB | 7.6M int (click/fav/cart/buy) |
| **IJCAI** | E-commerce | 423.4K / — / 874.3K | FatsMB | 3.6e7 int; largest |
| **Polyvore-U** | Fashion outfits | variable | Prompt-to-Slate | Polyvore Outfits variant (SIGIR'17) |
| **Steam** | Gaming bundles | 2.6M / ~615 / 15.5K | candidate (`additional_datasets.md`) | 7.8M reviews; publisher-curated bundles + text |

---

## 3. Key Observations

1. **Canonical trio = iFashion + Youshu + NetEase.** Appear together in all core diffusion/contrastive bundle papers (RDiffBR, MoDiffE, DDMMCL, DCBR). Beating all three = table stakes for a bundle-rec submission.
2. **Density / bundle-size spread is wide.** B–I density 0.01% (iFashion) → 0.77% (MealRec+H); avg bundle size 3 → 78 (NetEase). Useful for stress-regime claims.
3. **MealRec+H/L** unique high-density food domain; only DCBR uses them for cold-start.
4. **Next-set ≠ static bundle.** TaoBao/JD/TaFeng (SNSRec) are temporal sets, distinct framing. Single-item sequential (ML-100K, Amazon*, Retailrocket) appear only in non-bundle diffusion/seqrec papers.
5. **Distribution-shift protocols are rare.** Only RDiffBR (ρ-drift grid) and DCBR (explicit cold-bundle split) deviate from plain 8:1:1 / 8:2 splits — this is the gap our drift-robustness work targets.
6. **Text-rich candidates noted but not deep-read** (`additional_datasets.md`): BundleRec (Sun et al. SIGIR'22), Spotify titles, Goodreads, Polyvore descriptions, Steam — flagged as "text-as-theme-proxy" sources.
