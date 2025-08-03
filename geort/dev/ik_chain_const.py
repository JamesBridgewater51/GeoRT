from ikpy.chain import Chain
from ikpy.link import OriginLink, URDFLink
from enum import Enum

# ----------------------------------------------------------------
# Thumb Finger Chain
# ----------------------------------------------------------------
R_thumb_chain = Chain(name='R_thumb', active_links_mask=[False, True, True, True, True, False, False], links=[
    OriginLink(),
    URDFLink(
      name="R_thumb_roll_joint",
      origin_translation=[-0.0075211, 0.022278, 0.040247],
      origin_orientation=[0, 0, 0],
      rotation=[1, 0, 0],
      bounds=(0, 1.21447)
    ),
    URDFLink(
      name="R_thumb_abad_joint",
      origin_translation=[0.035071, 0.0031679, -0.00092914],
      origin_orientation=[-0.20341, 0.20144, 0.77017],
      rotation=[0, 0, 1],
      bounds=(-1.385, 0)
    ),
    URDFLink(
      name="R_thumb_mcp_joint",
      origin_translation=[0.026935, 1.7509e-05, 0.0068611],
      origin_orientation=[0 ,-0.63678 ,0],
      rotation=[0, 1, 0],
      bounds=(-0.8312, 0)
    ),
    URDFLink(
      name="R_thumb_pip_joint",
      origin_translation=[0.055705, -4.9999e-05, 0.0076659],
      origin_orientation=[0,0,0],
      rotation=[0, 1, 0],
      bounds=(-1.3, 0.0)
    ),
    URDFLink(
      name="R_thumb_dip_joint",
      origin_translation=[0.034552, 0.00055, -0.00084913],
      origin_orientation=[0, 0, 0],
      rotation=[0, 1, 0],
      bounds=(-1.395, 0.0)
    ),
    URDFLink(
        name="R_thumb_tips",
        origin_translation=[0.031823, -0.00050001, -0.0052464],
        origin_orientation=[0, 0, 0],
        joint_type="fixed",
    )
])

# ----------------------------------------------------------------
# Index Finger Chain
# ----------------------------------------------------------------
R_index_chain = Chain(name='R_index', active_links_mask=[False, True, True, True, False, False], links=[
    OriginLink(),
    URDFLink(
        name="R_index_abad_joint",
        origin_translation=[-0.0103, 0.030778, 0.11935],
        origin_orientation=[0, 0, 0],
        rotation=[1, 0, 0],
        bounds=(-0.26, 0.26)
    ),
    URDFLink(
        name="R_index_mcp_joint",
        origin_translation=[0.01047, 0, 0],
        origin_orientation=[0, 0, 0],
        rotation=[0, 1, 0],
        bounds=(0,1.5),
    ),
    URDFLink(
        name="R_index_pip_joint",
        origin_translation=[-0.0039371, 0, 0.03983],
        origin_orientation=[0, 0, 0],
        rotation=[0, 1, 0],
        bounds=(0, 1.57)
    ),
    URDFLink(
        name="R_index_dip_joint",
        origin_translation=[0.0057881, 0, 0.026492],
        origin_orientation=[0, 0, 0],
        rotation=[0, 1, 0],
        bounds=(0, 1.28)
    ),
    URDFLink(
        name="R_index_tip",
        origin_translation=[-0.013777, 0, 0.02493],
        origin_orientation=[0, 0, 0],
        joint_type="fixed",
    )
])

# ----------------------------------------------------------------
# Middle Finger Chain
# ----------------------------------------------------------------
R_middle_chain = Chain(name='R_middle', active_links_mask=[False, True, True, True, False, False], links=[
    OriginLink(),
    URDFLink(
        name="R_middle_abad_joint",
        origin_translation=[-0.0103, 0.0066607, 0.11935],
        origin_orientation=[0, 0, 0],
        rotation=[1, 0, 0],
        bounds=(-0.26, 0.26),
    ),
    URDFLink(
        name="R_middle_mcp_joint",
        origin_translation=[0.01047, 0, -2.2703e-05],
        origin_orientation=[0, 0, 0],
        rotation=[0, 1, 0],
        bounds=(0, 1.49),
    ),
    URDFLink(
        name="R_middle_pip_joint",
        origin_translation=[-0.0041959, 0, 0.044846],
        origin_orientation=[0, 0, 0],
        rotation=[0, 1, 0],
        bounds=(0, 1.57),
    ),
    URDFLink(
        name="R_middle_dip_joint",
        origin_translation=[0.0035929, 1.3238e-05, 0.031617],
        origin_orientation=[0, 0, 0],
        rotation=[0, 1, 0],
        bounds=(0, 1.45),
    ),
    URDFLink(
        name="R_middle_tip",
        origin_translation=[-0.015456, 0, 0.023925],
        origin_orientation=[0, 0, 0],
        joint_type="fixed",
    )
])

# ----------------------------------------------------------------
# Ring Finger Chain
# ----------------------------------------------------------------
R_ring_chain = Chain(name='R_ring', active_links_mask=[False, True, True, True, False], links=[
    OriginLink(),
    URDFLink(
        name="R_ring_mcp_joint",
        origin_translation=[0.0001598, -0.013559, 0.11971],
        origin_orientation=[0, 0, 0],
        rotation=[0, 1, 0],
        bounds=(0, 1.5583)
    ),
    URDFLink(
        name="R_ring_pip_joint",
        origin_translation=[-0.0039295, 0, 0.03983],
        origin_orientation=[0, 0, 0],
        rotation=[0, 1, 0],
        bounds=(0, 1.37758)
    ),
    URDFLink(
        name="R_ring_dip_joint",
        origin_translation=[0.00706, 0, 0.026181],
        origin_orientation=[0, 0, 0],
        rotation=[0, 1, 0],
        bounds=(0, 1.2825),
    ),
    URDFLink(
        name="R_ring_tip_joint",
        origin_translation=[-0.011211, 0, 0.026185],
        origin_orientation=[0, 0, 0],
        joint_type="fixed",
    )
])

# ----------------------------------------------------------------
# Pinky Finger Chain
# ----------------------------------------------------------------
R_pinky_chain = Chain(name='R_pinky', active_links_mask=[False, True, True, True, False], links=[
    OriginLink(),
    URDFLink(
        name="R_pinky_mcp_joint",
        origin_translation=[0.0001598, -0.033617, 0.11271],
        origin_orientation=[0, 0, 0],
        rotation=[0, 1, 0],
        bounds=(0, 1.5583)
    ),
    URDFLink(
        name="R_pinky_pip_joint",
        origin_translation=[-0.0039295, 0, 0.03983],
        origin_orientation=[0, 0, 0],
        rotation=[0, 1, 0],
        bounds=(0, 1.37758)
    ),
    URDFLink(
        name="R_pinky_dip_joint",
        origin_translation=[0.00706, 0, 0.026181],
        origin_orientation=[0, 0, 0],
        rotation=[0, 1, 0],
        bounds=(0, 1.2825)
    ),
    URDFLink(
        name="R_pinky_tip_joint",
        origin_translation=[-0.011211, 0, 0.026185],
        origin_orientation=[0, 0, 0],
        joint_type="fixed",
    )
])

class R_chains(Enum):
    """
    Enum for the right hand chains.
    """
    R_thumb = R_thumb_chain
    R_index = R_index_chain
    R_middle = R_middle_chain
    R_ring = R_ring_chain
    R_pinky = R_pinky_chain

if __name__ == "__main__":
    # You can now use these chain objects
    print("--- Index Finger ---")
    print(R_index_chain)
    print("\n--- Middle Finger ---")
    print(R_middle_chain)
    print("\n--- Ring Finger ---")
    print(R_ring_chain)
    print("\n--- Pinky Finger ---")
    print(R_pinky_chain)

    vis_dbg = True
    if vis_dbg:
        import matplotlib.pyplot
        from mpl_toolkits.mplot3d import Axes3D
        ax = matplotlib.pyplot.figure().add_subplot(111, projection='3d')
        R_thumb_chain.plot(joints=[0.34, 0.0, -0.33, -0.375], ax=ax)
        R_index_chain.plot(joints=[0.0, 0.0, 0.0], ax=ax)
        R_middle_chain.plot(joints=[0.0, 0.0, 0.0], ax=ax)
        R_ring_chain.plot(joints=[0.0, 0.0], ax=ax)
        R_pinky_chain.plot(joints=[0.0, 0.0], ax=ax)
        matplotlib.pyplot.show()

