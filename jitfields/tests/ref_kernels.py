class kernels2:

    membrane = [
        [ 0, -1,  0],
        [-1, +4, -1],
        [ 0, -1,  0],
    ]

    bending = [
        [0,  0,  1,  0,  0],
        [0,  2, -8,  2,  0],
        [1, -8, 20, -8,  1],
        [0,  2, -8,  2,  0],
        [0,  0,  1,  0,  0],
    ]

    shears = [

        [[[ 0.  , -2.  ,  0.  ],
          [-1.  ,  6.  , -1.  ],
          [ 0.  , -2.  ,  0.  ]],

         [[-0.25,  0.  , +0.25],
          [ 0.,    0.  ,  0.  ],
          [+0.25,  0.  , -0.25]]],

        [[[-0.25,  0.  , +0.25],
          [ 0.  ,  0.  ,  0.  ],
          [+0.25,  0.  , -0.25]],

         [[ 0.  , -1.  ,  0.  ],
          [-2.  ,  6.  , -2.  ],
          [ 0.  , -1.  ,  0.  ]]]

    ]

    div = [

        [[[ 0.  , -1.  ,  0.  ],
          [ 0.  ,  2.  ,  0.  ],
          [ 0.  , -1.  ,  0.  ]],

         [[-0.25,  0.  , +0.25],
          [ 0.  ,  0.  ,  0.  ],
          [+0.25,  0.  , -0.25]]],

        [[[-0.25,  0.  , +0.25],
          [ 0.  ,  0.  ,  0.  ],
          [+0.25,  0.  , -0.25]],

         [[ 0.  ,  0.  ,  0.  ],
          [-1.  ,  2.  , -1.  ],
          [ 0.  ,  0.  ,  0.  ]]]

    ]


class kernels3:

    membrane = [

        [[ 0,  0,  0],
         [ 0, -1,  0],
         [ 0,  0,  0]],

        [[ 0, -1,  0],
         [-1,  6, -1],
         [ 0, -1,  0]],

        [[ 0,  0,  0],
         [ 0, -1,  0],
         [ 0,  0,  0]],

    ]

    bending = [
        [[0,   0,   0,   0,   0],
         [0,   0,   0,   0,   0],
         [0,   0,   1,   0,   0],
         [0,   0,   0,   0,   0],
         [0,   0,   0,   0,   0]],

        [[0,   0,   0,   0,   0],
         [0,   0,   2,   0,   0],
         [0,   2, -12,   2,   0],
         [0,   0,   2,   0,   0],
         [0,   0,   0,   0,   0]],

        [[0,   0,   1,   0,   0],
         [0,   2, -12,   2,   0],
         [1, -12, +42, -12,   1],
         [0,   2, -12,   2,   0],
         [0,   0,   1,   0,   0]],

        [[0,   0,   0,   0,   0],
         [0,   0,   2,   0,   0],
         [0,   2, -12,   2,   0],
         [0,   0,   2,   0,   0],
         [0,   0,   0,   0,   0]],

        [[0,   0,   0,   0,   0],
         [0,   0,   0,   0,   0],
         [0,   0,   1,   0,   0],
         [0,   0,   0,   0,   0],
         [0,   0,   0,   0,   0]]
    ]

    shears = [
        # XX
        [[[[ 0.0000,  0.0000,  0.0000],
           [ 0.0000, -2.0000,  0.0000],
           [ 0.0000,  0.0000,  0.0000]],
          [[ 0.0000, -1.0000,  0.0000],
           [-1.0000,  8.0000, -1.0000],
           [ 0.0000, -1.0000,  0.0000]],
          [[ 0.0000,  0.0000,  0.0000],
           [ 0.0000, -2.0000,  0.0000],
           [ 0.0000,  0.0000,  0.0000]]],
         # XY
         [[[ 0.0000, -0.2500,  0.0000],
           [ 0.0000,  0.0000,  0.0000],
           [ 0.0000,  0.2500,  0.0000]],
          [[ 0.0000,  0.0000,  0.0000],
           [ 0.0000,  0.0000,  0.0000],
           [ 0.0000,  0.0000,  0.0000]],
          [[ 0.0000,  0.2500,  0.0000],
           [ 0.0000,  0.0000,  0.0000],
           [ 0.0000, -0.2500,  0.0000]]],
         # XZ
         [[[ 0.0000,  0.0000,  0.0000],
           [-0.2500,  0.0000,  0.2500],
           [ 0.0000,  0.0000,  0.0000]],
          [[ 0.0000,  0.0000,  0.0000],
           [ 0.0000,  0.0000,  0.0000],
           [ 0.0000,  0.0000,  0.0000]],
          [[ 0.0000,  0.0000,  0.0000],
           [ 0.2500,  0.0000, -0.2500],
           [ 0.0000,  0.0000,  0.0000]]]],
        # YX
        [[[[ 0.0000, -0.2500,  0.0000],
           [ 0.0000,  0.0000,  0.0000],
           [ 0.0000,  0.2500,  0.0000]],
          [[ 0.0000,  0.0000,  0.0000],
           [ 0.0000,  0.0000,  0.0000],
           [ 0.0000,  0.0000,  0.0000]],
          [[ 0.0000,  0.2500,  0.0000],
           [ 0.0000,  0.0000,  0.0000],
           [ 0.0000, -0.2500,  0.0000]]],
         # YY
         [[[ 0.0000,  0.0000,  0.0000],
           [ 0.0000, -1.0000,  0.0000],
           [ 0.0000,  0.0000,  0.0000]],
          [[ 0.0000, -2.0000,  0.0000],
           [-1.0000,  8.0000, -1.0000],
           [ 0.0000, -2.0000,  0.0000]],
          [[ 0.0000,  0.0000,  0.0000],
           [ 0.0000, -1.0000,  0.0000],
           [ 0.0000,  0.0000,  0.0000]]],
         # YZ
         [[[ 0.0000,  0.0000,  0.0000],
           [ 0.0000,  0.0000,  0.0000],
           [ 0.0000,  0.0000,  0.0000]],
          [[-0.2500,  0.0000,  0.2500],
           [ 0.0000,  0.0000,  0.0000],
           [ 0.2500,  0.0000, -0.2500]],
          [[ 0.0000,  0.0000,  0.0000],
           [ 0.0000,  0.0000,  0.0000],
           [ 0.0000,  0.0000,  0.0000]]]],
        # ZX
        [[[[ 0.0000,  0.0000,  0.0000],
           [-0.2500,  0.0000,  0.2500],
           [ 0.0000,  0.0000,  0.0000]],
          [[ 0.0000,  0.0000,  0.0000],
           [ 0.0000,  0.0000,  0.0000],
           [ 0.0000,  0.0000,  0.0000]],
          [[ 0.0000,  0.0000,  0.0000],
           [ 0.2500,  0.0000, -0.2500],
           [ 0.0000,  0.0000,  0.0000]]],
         # ZY
         [[[ 0.0000,  0.0000,  0.0000],
           [ 0.0000,  0.0000,  0.0000],
           [ 0.0000,  0.0000,  0.0000]],
          [[-0.2500,  0.0000,  0.2500],
           [ 0.0000,  0.0000,  0.0000],
           [ 0.2500,  0.0000, -0.2500]],
          [[ 0.0000,  0.0000,  0.0000],
           [ 0.0000,  0.0000,  0.0000],
           [ 0.0000,  0.0000,  0.0000]]],
         # ZZ
         [[[ 0.0000,  0.0000,  0.0000],
           [ 0.0000, -1.0000,  0.0000],
           [ 0.0000,  0.0000,  0.0000]],
          [[ 0.0000, -1.0000,  0.0000],
           [-2.0000,  8.0000, -2.0000],
           [ 0.0000, -1.0000,  0.0000]],
          [[ 0.0000,  0.0000,  0.0000],
           [ 0.0000, -1.0000,  0.0000],
           [ 0.0000,  0.0000,  0.0000]]]]
    ]

    div = [
        # XX
        [[[[ 0.0000,  0.0000,  0.0000],
           [ 0.0000, -1.0000,  0.0000],
           [ 0.0000,  0.0000,  0.0000]],
          [[ 0.0000, -0.0000,  0.0000],
           [-0.0000,  2.0000, -0.0000],
           [ 0.0000, -0.0000,  0.0000]],
          [[ 0.0000,  0.0000,  0.0000],
           [ 0.0000, -1.0000,  0.0000],
           [ 0.0000,  0.0000,  0.0000]]],
         # XY
         [[[ 0.0000, -0.2500,  0.0000],
           [ 0.0000,  0.0000,  0.0000],
           [ 0.0000,  0.2500,  0.0000]],
          [[ 0.0000,  0.0000,  0.0000],
           [ 0.0000,  0.0000,  0.0000],
           [ 0.0000,  0.0000,  0.0000]],
          [[ 0.0000,  0.2500,  0.0000],
           [ 0.0000,  0.0000,  0.0000],
           [ 0.0000, -0.2500,  0.0000]]],
         # XZ
         [[[ 0.0000,  0.0000,  0.0000],
           [-0.2500,  0.0000,  0.2500],
           [ 0.0000,  0.0000,  0.0000]],
          [[ 0.0000,  0.0000,  0.0000],
           [ 0.0000,  0.0000,  0.0000],
           [ 0.0000,  0.0000,  0.0000]],
          [[ 0.0000,  0.0000,  0.0000],
           [ 0.2500,  0.0000, -0.2500],
           [ 0.0000,  0.0000,  0.0000]]]],
        # YX
        [[[[ 0.0000, -0.2500,  0.0000],
           [ 0.0000,  0.0000,  0.0000],
           [ 0.0000,  0.2500,  0.0000]],
          [[ 0.0000,  0.0000,  0.0000],
           [ 0.0000,  0.0000,  0.0000],
           [ 0.0000,  0.0000,  0.0000]],
          [[ 0.0000,  0.2500,  0.0000],
           [ 0.0000,  0.0000,  0.0000],
           [ 0.0000, -0.2500,  0.0000]]],
         # YY
         [[[ 0.0000,  0.0000,  0.0000],
           [ 0.0000, -0.0000,  0.0000],
           [ 0.0000,  0.0000,  0.0000]],
          [[ 0.0000, -1.0000,  0.0000],
           [-0.0000,  2.0000, -0.0000],
           [ 0.0000, -1.0000,  0.0000]],
          [[ 0.0000,  0.0000,  0.0000],
           [ 0.0000, -0.0000,  0.0000],
           [ 0.0000,  0.0000,  0.0000]]],
         # YZ
         [[[ 0.0000,  0.0000,  0.0000],
           [ 0.0000,  0.0000,  0.0000],
           [ 0.0000,  0.0000,  0.0000]],
          [[-0.2500,  0.0000,  0.2500],
           [ 0.0000,  0.0000,  0.0000],
           [ 0.2500,  0.0000, -0.2500]],
          [[ 0.0000,  0.0000,  0.0000],
           [ 0.0000,  0.0000,  0.0000],
           [ 0.0000,  0.0000,  0.0000]]]],
        # ZX
        [[[[ 0.0000,  0.0000,  0.0000],
           [-0.2500,  0.0000,  0.2500],
           [ 0.0000,  0.0000,  0.0000]],
          [[ 0.0000,  0.0000,  0.0000],
           [ 0.0000,  0.0000,  0.0000],
           [ 0.0000,  0.0000,  0.0000]],
          [[ 0.0000,  0.0000,  0.0000],
           [ 0.2500,  0.0000, -0.2500],
           [ 0.0000,  0.0000,  0.0000]]],
         # ZY
         [[[ 0.0000,  0.0000,  0.0000],
           [ 0.0000,  0.0000,  0.0000],
           [ 0.0000,  0.0000,  0.0000]],
          [[-0.2500,  0.0000,  0.2500],
           [ 0.0000,  0.0000,  0.0000],
           [ 0.2500,  0.0000, -0.2500]],
          [[ 0.0000,  0.0000,  0.0000],
           [ 0.0000,  0.0000,  0.0000],
           [ 0.0000,  0.0000,  0.0000]]],
         # ZZ
         [[[ 0.0000,  0.0000,  0.0000],
           [ 0.0000, -0.0000,  0.0000],
           [ 0.0000,  0.0000,  0.0000]],
          [[ 0.0000, -0.0000,  0.0000],
           [-1.0000,  2.0000, -1.0000],
           [ 0.0000, -0.0000,  0.0000]],
          [[ 0.0000,  0.0000,  0.0000],
           [ 0.0000, -0.0000,  0.0000],
           [ 0.0000,  0.0000,  0.0000]]]]
    ]
