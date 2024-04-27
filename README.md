# Improved Methods for Consistency Training in JAX

[![arXiv](https://img.shields.io/badge/arXiv-2310.14189-b31b1b.svg)](https://arxiv.org/abs/2310.14189)
<a target="_blank" href="https://colab.research.google.com/github/leakedweights/mincy/blob/main/notebooks/colab.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

under construction

@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@&&&&&&&@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@&&&&&&&&@@@@@@@
@@@@BGGGBBBBBGBBGBB#B&&@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@&&#B#BBBGB##BBBBGG#@@@@
@@@PPGB#####&&&&###BGGPGG#&@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@&&BGPPPGB###&&&&#&###BGPG@@@
@@PPBB##&&&&&&##GP55YJJJJY5PG#&@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@&#BPYYJJJ?JYYY5GB#&&&&&&&#BBPB@@
@B5B######&&#BP5YYJJJJJJJ??JPGGPB&@@&GG#@@@@@@@@@@@@@@@@@@@#GG&@@&#BGBB5J??JJJJJYYYY5PG##&######B5@@
@BPBB##&&&BP5JJYYJYYYYYYJYGB##&#BGGB#@&GP#@@@@@@@@@@@@@@@#PG&@@#GGB#&&&##G5YYYYYYYJJYJJY5B&&&###B5#@
@BGB#####GYJJJJJJJYYYYYPG#&&&&&&&&#BGG#@&BP#@@@@@@@@@@@#PG&@&BGB##&&&&&&&&#BPYYYYYYJJJYJJYP#####BPG@
@BGB####GYJJJYYJYYJYY5B&&&&&&&&&&&&&#BGG#&&GP#@@@@@@@&5G&@BGPGB#&&&&&&&&&&&&&B5JJJYJJYJJJJYG####B5B@
@BPB##BPYJJJJJJJJJJ5G#&&&&&&&&&&&#BP5Y5555G##PP&@@@@BP#&G5YYYYYY5G#&&&&&&&&&&&#G5JJJJJJJJJJY5B##B5&@
@&5BBPYJJYYYYYYJJ5G&&&&&&&&&&&&&B5YYYYYYYYYYP##PG&#GB@BYYYYYYYYYJY5G#&&&&&&&&&&&&B5JJYYYYYYJJYPBBP@@
@@GGPJJJJJJJJJJYG#&&&&&&&&&&&&BP5YYYYYYYYYY555G#PGGB&P55YYYYJYYYYYY5PB#&&&&&&&&&&&#G5YJJJJJJJJJPP#@@
@@&5J???JJYY5PB#&&&&&&&&&&&#BPYJJJYYYYY555Y55PPP##&#5PP5YYYYYYYYJJJJJY5G#&&&&&&&&&&&&#GPYYJ????J5@@@
@@@BJ???JJ5G#&&&&&&&&&&&&&#PYJJJJJJJYYYYYY55PGB##B#&BBGP5YYYYYYYJJJJJJJJ5B&&&&&&&&&&&&##BPYJJ??Y&@@@
@@@@BY5PGB######&&&&&&##BBPYYYYYY5555555555PGGB&B7P#&GPP5555Y555555Y55YYYPGBB#&&&&&&######BGG55&@@@@
@@@@@&BGGBBBBBBBB##BBBBGGGGGGGGPYYYYJYYYYY55PGG#P5GB#GGP55YYYYJJJYYYPGBGGGGGGBBGGBBBBBBBBGGPPG@@@@@@
@@@@@@@@&&&&&#&&&#P5GB##BB###BP5YJJJYYJYYY55PGB######BG555YYYJYYJJJY5PB###BB##BPPG#&#######&&@@@@@@@
@@@@@@@@@@@@@@@BGPB######&#BGP5YYJJYYJJYYY5PGBB#&&###BBG5YYYJJJYYJJYY55PB#&&&####BBG#@@@@@@@@@@@@@@@
@@@@@@@@@@@@@#GGB##&&&&&#BP555YJJJYYJJYYY5GB#BGG##BBGPG#BG5YYYJJYYJJJYY5YPGB&&&&&##BGB&@@@@@@@@@@@@@
@@@@@@@@@@@&PGB##&&&&#G5JJJYYJJJYYYYYYY5G###GY5PBBGG5YYG###G5YYYYYJYJJJYJJJJ5PB&&&###BGG@@@@@@@@@@@@
@@@@@@@@@@@GG###&&&BPYJJYYYYYYYJYYYY55G#&&BG5Y5@&#PB@YJYPB#&#G5YYYYYJJYYYYYYJJYPB&&&###GB@@@@@@@@@@@
@@@@@@@@@@BP####&&#5YYYYYYYYYJYYY555P#&&&BG5Y5@@##G&@#YY5PB#&&#G5P5YYYJYYYYYYYYYP#&&####P&@@@@@@@@@@
@@@@@@@@@@PB###&&#GYYYYYYJYYYY55PGG#&&&#BGP55&@@#BP&@@#55PGB#&&&#BGP5YYYYYJYYYYYYP#&####G&@@@@@@@@@@
@@@@@@@@@@BG#####PYYYYYYYYYJY5PB#&&&&&#BBGG5&@@@BPG@@@@#PGGB##&&&&&##G5JJJYYYYYYYYP#####G&@@@@@@@@@@
@@@@@@@@@@@GB##&#BP5YYYYYJJJP#&&&&&&&###BBP&@@@@@#&@@@@@BGBB###&&&&&&&#PYJJJJYYY5G#&###GB@@@@@@@@@@@
@@@@@@@@@@@@#G##&##BG5JJJYP#&&&&&&&&##B#BG#@@@@@@@@@@@@@@BGBBB#&&&&&&&&&#GYYYY5B#&&&##P#@@@@@@@@@@@@
@@@@@@@@@@@@@BG##&&&#BBBB#&&&&&&&&&&###BG&@@@@@@@@@@@@@@@@#PB###&&&&&&&&&&&####&&&&##P#@@@@@@@@@@@@@
@@@@@@@@@@@@@@GGB##&&&&&&&&&&&&&&&&#BGGG@@@@@@@@@@@@@@@@@@@&BPGB##&##&&&&###&&&&#BBG5P@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@&&&#GB####BGPP##55PPBGG&@@@@@@@@@@@@@@@@@@@@@@&BPG555P##PPGB###BGP#&&&@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@&GGPGBBGGBBGGP5PB&@@@@@@@@@@@@@@@@@@@@@@@@@@&B55PGPGGGGGGP55G@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@&&@&#####&&@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@&&#BB#BB@@@&@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@