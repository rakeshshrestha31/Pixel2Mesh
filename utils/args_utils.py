##
#  @author Rakesh Shrestha, rakeshs@sfu.ca

## adds arugments related to neural network model
def add_model_args(parser):
    parser.add_argument('--gconv-skip-connection',
                        help='[none|add|concat]', type=str)
    parser.add_argument('--mvsnet-features-list', nargs='+', type=int)

    parser.add_argument('--use-rgb-features', dest='use_rgb_features',
                        action='store_true')
    parser.add_argument('--dont-use-rgb-features', dest='use_rgb_features',
                        action='store_false')

    parser.add_argument('--use-costvolume-features', dest='use_costvolume_features',
                        action='store_true')
    parser.add_argument('--dont-use-costvolume-features', dest='use_costvolume_features',
                        action='store_false')

    parser.add_argument('--use-depth-features', dest='use_depth_features',
                        action='store_true')
    parser.add_argument('--dont-use-depth-features', dest='use_depth_features',
                        action='store_false')

    parser.add_argument('--use-contrastive-depth', dest='use_contrastive_depth',
                        action='store_true')
    parser.add_argument('--dont-use-contrastive-depth', dest='use_contrastive_depth',
                        action='store_false')

    parser.add_argument('--use-predicted-depth-as-feature',
                        dest='use_predicted_depth_as_feature',
                        action='store_true')
    parser.add_argument('--dont-use-predicted-depth-as-feature',
                        dest='use_predicted_depth_as_feature',
                        action='store_false')

    parser.add_argument('--use-backprojected-depth-as-feature',
                        dest='use_backprojected_depth_as_feature',
                        action='store_true')
    parser.add_argument('--dont-use-backprojected-depth-as-feature',
                        dest='use_backprojected_depth_as_feature',
                        action='store_false')

    parser.add_argument('--use-multiview-coords-as-feature',
                        dest='use_multiview_coords_as_feature',
                        action='store_true')
    parser.add_argument('--dont-use-multiview-coords-as-feature',
                        dest='use_multiview_coords_as_feature',
                        action='store_false')

    parser.add_argument('--use-stats-query-attention',
                        dest='use_stats_query_attention',
                        help='use statistical features '
                             'for attention query vector',
                        action='store_true')
    parser.add_argument('--dont-use-stats-query-attention',
                        dest='use_stats_query_attention',
                        help='don\'t use statistical features '
                             'for attention query vector',
                        action='store_false')

    parser.add_argument('--feature-fusion-method',
                        help='[concat|stats|attention]', type=str)
    parser.add_argument('--num-attention-heads',
                        help='number of attention heads', type=int)
    parser.add_argument('--num-attention-features',
                        help='number of attention features '
                             '< 0 indicates no change in features number '
                             'from input',
                        type=int)

    # set the default booleans to None otherwise it will be False and
    # overwrite the default options
    parser.set_defaults(
        use_rgb_features=None,
        use_costvolume_features=None,
        use_depth_features=None,
        use_contrastive_depth=None,
        use_predicted_depth_as_feature=None,
        use_backprojected_depth_as_feature=None,
        use_multiview_coords_as_feature=None,
        use_stats_query_attention=None
    )

