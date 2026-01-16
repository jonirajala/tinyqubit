"""
Pattern-based gate optimization.

Declarative rules applied until fixed point:
    - Cancellation: [X,X]→[], [H,H]→[], [CX,CX]→[]
    - Merge: [RZ(a),RZ(b)]→[RZ(a+b)]
    - Deterministic rule application order
"""
