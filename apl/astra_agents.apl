⍝ SPDX-License-Identifier: LicenseRef-NON-AI-MPL-2.0
⍝ Copyright (C) 2026 SnapKitty Collective
⍝ Astra APL: agent registry and numerical specialist implementations.
⍝ An agent receives a nested request: numeric-data prior-proposals round.
⍝ A proposal is a numeric vector; specialists may implement any APL function
⍝ with this calling convention. These built-ins compute real estimators.

∇ ResetAgents
 AgentProfiles←⍬
 AgentFunctions←⍬
∇

∇ Z←Register X;P;F;I
 ⍝ Input: profile function-name. Return index, or zero when rejected.
 Z←0
 →(2≠⍴X)/0
 P←⊃X[1]
 F←⊃X[2]
 →(5≠⍴P)/0
 →(3≠⎕NC F)/0
 I←1
LOOP:→(I>⍴AgentProfiles)/ADD
 →((⊃P[1])≡⊃(⊃AgentProfiles[I])[1])/0
 I←I+1
 →LOOP
ADD:AgentProfiles←AgentProfiles,⊂P
 AgentFunctions←AgentFunctions,⊂F
 Z←⍴AgentProfiles
∇

∇ Z←MeanAgent Request;X
 X←⊃Request[1]
 Z←⍬
 →(0=⍴X)/0
 Z←,(+/X)÷⍴X
∇

∇ Z←MedianAgent Request;X;N;S
 X←⊃Request[1]
 Z←⍬
 N←⍴X
 →(N=0)/0
 S←X[⍋X]
 Z←,0.5×S[⌈N÷2]+S[1+⌊N÷2]
∇

∇ Z←TrimmedAgent Request;X;N;Drop;S
 X←⊃Request[1]
 Z←⍬
 N←⍴X
 →(N=0)/0
 Drop←⌊N÷5
 S←X[⍋X]
 S←Drop↓(-Drop)↓S
 Z←,(+/S)÷⍴S
∇

∇ Z←MidrangeAgent Request;X
 X←⊃Request[1]
 Z←⍬
 →(0=⍴X)/0
 Z←,0.5×(⌊/X)+⌈/X
∇

∇ Z←WinsorAgent Request;X;N;S;D;Low;High
 X←⊃Request[1]
 Z←⍬
 N←⍴X
 →(N=0)/0
 S←X[⍋X]
 D←⌊N÷5
 Low←S[1+D]
 High←S[N-D]
 S←Low⌈High⌊X
 Z←,(+/S)÷N
∇

∇ Z←ConsensusAgent Request;X;Prior;I;Values
 X←⊃Request[1]
 Prior←⊃Request[2]
 →(0=⍴Prior)/FIRST
 Values←⍬
 I←1
LOOP:→(I>⍴Prior)/DONE
 Values←Values,⊃Prior[I]
 I←I+1
 →LOOP
DONE:Z←MedianAgent (⊂Values),(⊂⍬),⊂1
 →0
FIRST:Z←MedianAgent Request
∇

∇ Z←WeightedMean X;Weights;Values;I;Width;P
 ⍝ Input: weights and nested proposals. Reject incompatible dimensions.
 Z←⍬
 Weights←⊃X[1]
 Values←⊃X[2]
 →(0=⍴Values)/0
 →((⍴Weights)≠⍴Values)/0
 →(∨/Weights<0)/0
 →((+/Weights)≤0)/0
 Width←⍴⊃Values[1]
 →(Width=0)/0
 P←Width⍴0
 I←1
LOOP:→(I>⍴Values)/DONE
 →(Width≠⍴⊃Values[I])/0
 P←P+Weights[I]×⊃Values[I]
 I←I+1
 →LOOP
DONE:Z←P÷+/Weights
∇
⍝ Astra APL: deterministic capability and budget routing.
⍝ All indices are one-origin. No external routing framework is used.
⍝ A profile is (name capabilities keywords cost enabled).

∇ Z←Lower X;U;L;I;M
 U←'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
 L←'abcdefghijklmnopqrstuvwxyz'
 Z←,X
 I←U⍳Z
 M←I≤⍴U
 Z[M/⍳⍴Z]←L[M/I]
∇

∇ Z←Tokens X;S;A;I;W
 S←Lower X
 A←'abcdefghijklmnopqrstuvwxyz0123456789_'
 Z←⍬
 W←''
 I←1
LOOP:→(I>⍴S)/END
 →(~S[I]∊A)/BREAK
 W←W,S[I]
 →NEXT
BREAK:→(0=⍴W)/NEXT
 Z←Z,⊂W
 W←''
NEXT:I←I+1
 →LOOP
END:→(0=⍴W)/0
 Z←Z,⊂W
∇

∇ Z←Unique X;I
 Z←⍬
 I←1
LOOP:→(I>⍴X)/0
 →(∨/X[I]∊Z)/NEXT
 Z←Z,X[I]
NEXT:I←I+1
 →LOOP
∇

∇ Z←Profile X;N;C;K;V;E
 ⍝ Construct a validated profile from a five-element nested vector.
 Z←⍬
 →(5≠⍴X)/0
 N←⊃X[1]
 C←⊃X[2]
 K←⊃X[3]
 V←⊃X[4]
 E←⊃X[5]
 →(0=⍴N)/0
 →(1≠≡V)/0
 →(V≤0)/0
 →(~E∊0 1)/0
 Z←(⊂N),(⊂Unique C),(⊂Unique Tokens K),(⊂V),⊂E
∇

∇ Z←Required Eligible P;Caps
 Caps←⊃P[2]
 Z←(⊃P[5])∧∧/Required∊Caps
∇

∇ Z←Normalize X;T
 Z←0×X
 →(0=⍴X)/0
 T←+/X
 →(T≤0)/0
 Z←X÷T
∇

∇ Z←Profiles Route Request;Text;Need;Want;K;Budget;Excluded;Query;Used;Covered;IDs;Scores;Spent;Matches;NewCaps;I;P;Cost;Score;Utility;Best;BestUtility;BestScore;BestCost;Tie;Weights
 ⍝ Request: prompt required preferred top-k budget excluded-indices.
 ⍝ Result: selected-indices weights raw-scores total-cost.
 Text←⊃Request[1]
 Need←⊃Request[2]
 Want←⊃Request[3]
 K←⊃Request[4]
 Budget←⊃Request[5]
 Excluded←⊃Request[6]
 Query←Unique Tokens Text
 Used←Excluded
 Covered←⍬
 IDs←⍬
 Scores←⍬
 Spent←0
 Z←(⊂IDs),(⊂IDs),(⊂IDs),⊂Spent
 →((K≤0)∨K≠⌊K)/0
 →(Budget≤0)/0
ROUND:→((⍴IDs)≥K)/DONE
 Best←0
 BestUtility←¯1
 BestScore←0
 BestCost←0
 I←1
SCAN:→(I>⍴Profiles)/CHOOSE
 →(I∊Used)/NEXT
 P←⊃Profiles[I]
 →(~Need Eligible P)/NEXT
 Cost←⊃P[4]
 →(Cost>Budget-Spent)/NEXT
 Matches←+/Query∊⊃P[3]
 NewCaps←((⊃P[2])∊Want)∧~(⊃P[2])∊Covered
 Score←1+(2×Matches÷1⌈⍴Query)+3×+/NewCaps
 Utility←Score÷Cost
 →(Utility>BestUtility)/KEEP
 →(Utility<BestUtility)/NEXT
 →(Cost<BestCost)/KEEP
 →(Cost>BestCost)/NEXT
 ⍝ Stable registration order is the final tie breaker.
 →NEXT
KEEP:Best←I
 BestUtility←Utility
 BestScore←Score
 BestCost←Cost
NEXT:I←I+1
 →SCAN
CHOOSE:→(Best=0)/DONE
 IDs←IDs,Best
 Scores←Scores,BestScore
 Used←Used,Best
 P←⊃Profiles[Best]
 Covered←Unique Covered,⊃P[2]
 Spent←Spent+BestCost
 →ROUND
DONE:Weights←Normalize Scores
 Z←(⊂IDs),(⊂Weights),(⊂Scores),⊂Spent
∇
⍝ Astra APL: bounded-round orchestration with budget accounting.
⍝ Execution is synchronous in GNU APL. No concurrency or preemption claim.

∇ Z←Function Invoke Request;Caught;Value
 ⍝ Function is a registered APL name, never text obtained from the prompt.
 Value←⍬
 Caught←⎕EC 'Value←',Function,' Request'
 Z←(⊂0),⊂⍬
 →(0=⊃Caught[1])/0
 →(0=⍴Value)/0
 Z←(⊂1),⊂Value
∇

∇ Z←Mixture Config;Prompt;Data;Need;Want;K;Budget;Rounds;Round;Spent;Excluded;Previous;History;Failures;Routes;Request;Decision;IDs;Weights;Good;GoodWeights;I;ID;Outcome;Function;Answer;Remaining
 ⍝ Config: prompt data required preferred top-k total-budget round-count.
 ⍝ Result: answer history routes failures spent success.
 Prompt←⊃Config[1]
 Data←⊃Config[2]
 Need←⊃Config[3]
 Want←⊃Config[4]
 K←⊃Config[5]
 Budget←⊃Config[6]
 Rounds←⊃Config[7]
 Round←1
 Spent←0
 Excluded←⍬
 Previous←⍬
 History←⍬
 Failures←⍬
 Routes←⍬
 Answer←⍬
 Z←(⊂Answer),(⊂History),(⊂Routes),(⊂Failures),(⊂Spent),⊂0
 →((Rounds≤0)∨Rounds≠⌊Rounds)/0
LOOP:→(Round>Rounds)/DONE
 Remaining←Budget-Spent
 →(Remaining≤0)/DONE
 Request←(⊂Prompt),(⊂Need),(⊂Want),(⊂K),(⊂Remaining),⊂Excluded
 Decision←AgentProfiles Route Request
 IDs←⊃Decision[1]
 Weights←⊃Decision[2]
 →(0=⍴IDs)/DONE
 Routes←Routes,⊂Decision
 Spent←Spent+⊃Decision[4]
 Good←⍬
 GoodWeights←⍬
 Request←(⊂Data),(⊂Previous),⊂Round
 I←1
AGENT:→(I>⍴IDs)/AGGREGATE
 ID←IDs[I]
 Function←⊃AgentFunctions[ID]
 Outcome←Function Invoke Request
 →(0=⊃Outcome[1])/FAIL
 Good←Good,Outcome[2]
 GoodWeights←GoodWeights,Weights[I]
 History←History,⊂(⊂Round),(⊂ID),Outcome[2]
 →NEXT
FAIL:Excluded←Unique Excluded,ID
 Failures←Failures,⊂Round ID
NEXT:I←I+1
 →AGENT
AGGREGATE:→(0=⍴Good)/ADVANCE
 Previous←Good
 Answer←WeightedMean (⊂GoodWeights),⊂Good
ADVANCE:Round←Round+1
 →LOOP
DONE:Z←(⊂Answer),(⊂History),(⊂Routes),(⊂Failures),(⊂Spent),⊂0<⍴Answer
∇
