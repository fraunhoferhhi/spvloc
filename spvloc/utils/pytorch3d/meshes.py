# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Sequence, Tuple, Union

import torch

Device = Union[str, torch.device]


class Meshes:
    """
    This class provides functions for working with batches of triangulated
    meshes with varying numbers of faces and vertices, and converting between
    representations.

    Within Meshes, there are three different representations of the faces and
    verts data:

    List
      - only used for input as a starting point to convert to other representations.
    Padded
      - has specific batch dimension.
    Packed
      - no batch dimension.
      - has auxiliary variables used to index into the padded representation.

    Example:

    Input list of verts V_n = [[V_1], [V_2], ... , [V_N]]
    where V_1, ... , V_N are the number of verts in each mesh and N is the
    number of meshes.

    Input list of faces F_n = [[F_1], [F_2], ... , [F_N]]
    where F_1, ... , F_N are the number of faces in each mesh.

    # SPHINX IGNORE
     List                      | Padded                  | Packed
    ---------------------------|-------------------------|------------------------
    [[V_1], ... , [V_N]]       | size = (N, max(V_n), 3) |  size = (sum(V_n), 3)
                               |                         |
    Example for verts:         |                         |
                               |                         |
    V_1 = 3, V_2 = 4, V_3 = 5  | size = (3, 5, 3)        |  size = (12, 3)
                               |                         |
    List([                     | tensor([                |  tensor([
      [                        |     [                   |    [0.1, 0.3, 0.5],
        [0.1, 0.3, 0.5],       |       [0.1, 0.3, 0.5],  |    [0.5, 0.2, 0.1],
        [0.5, 0.2, 0.1],       |       [0.5, 0.2, 0.1],  |    [0.6, 0.8, 0.7],
        [0.6, 0.8, 0.7],       |       [0.6, 0.8, 0.7],  |    [0.1, 0.3, 0.3],
      ],                       |       [0,    0,    0],  |    [0.6, 0.7, 0.8],
      [                        |       [0,    0,    0],  |    [0.2, 0.3, 0.4],
        [0.1, 0.3, 0.3],       |     ],                  |    [0.1, 0.5, 0.3],
        [0.6, 0.7, 0.8],       |     [                   |    [0.7, 0.3, 0.6],
        [0.2, 0.3, 0.4],       |       [0.1, 0.3, 0.3],  |    [0.2, 0.4, 0.8],
        [0.1, 0.5, 0.3],       |       [0.6, 0.7, 0.8],  |    [0.9, 0.5, 0.2],
      ],                       |       [0.2, 0.3, 0.4],  |    [0.2, 0.3, 0.4],
      [                        |       [0.1, 0.5, 0.3],  |    [0.9, 0.3, 0.8],
        [0.7, 0.3, 0.6],       |       [0,    0,    0],  |  ])
        [0.2, 0.4, 0.8],       |     ],                  |
        [0.9, 0.5, 0.2],       |     [                   |
        [0.2, 0.3, 0.4],       |       [0.7, 0.3, 0.6],  |
        [0.9, 0.3, 0.8],       |       [0.2, 0.4, 0.8],  |
      ]                        |       [0.9, 0.5, 0.2],  |
    ])                         |       [0.2, 0.3, 0.4],  |
                               |       [0.9, 0.3, 0.8],  |
                               |     ]                   |
                               |  ])                     |
    Example for faces:         |                         |
                               |                         |
    F_1 = 1, F_2 = 2, F_3 = 7  | size = (3, 7, 3)        | size = (10, 3)
                               |                         |
    List([                     | tensor([                | tensor([
      [                        |     [                   |    [ 0,  1,  2],
        [0, 1, 2],             |       [0,   1,  2],     |    [ 3,  4,  5],
      ],                       |       [-1, -1, -1],     |    [ 4,  5,  6],
      [                        |       [-1, -1, -1]      |    [ 8,  9,  7],
        [0, 1, 2],             |       [-1, -1, -1]      |    [ 7,  8, 10],
        [1, 2, 3],             |       [-1, -1, -1]      |    [ 9, 10,  8],
      ],                       |       [-1, -1, -1],     |    [11, 10,  9],
      [                        |       [-1, -1, -1],     |    [11,  7,  8],
        [1, 2, 0],             |     ],                  |    [11, 10,  8],
        [0, 1, 3],             |     [                   |    [11,  9,  8],
        [2, 3, 1],             |       [0,   1,  2],     |  ])
        [4, 3, 2],             |       [1,   2,  3],     |
        [4, 0, 1],             |       [-1, -1, -1],     |
        [4, 3, 1],             |       [-1, -1, -1],     |
        [4, 2, 1],             |       [-1, -1, -1],     |
      ],                       |       [-1, -1, -1],     |
    ])                         |       [-1, -1, -1],     |
                               |     ],                  |
                               |     [                   |
                               |       [1,   2,  0],     |
                               |       [0,   1,  3],     |
                               |       [2,   3,  1],     |
                               |       [4,   3,  2],     |
                               |       [4,   0,  1],     |
                               |       [4,   3,  1],     |
                               |       [4,   2,  1],     |
                               |     ]                   |
                               |   ])                    |
    -----------------------------------------------------------------------------

    Auxiliary variables for packed representation

    Name                           |   Size              |  Example from above
    -------------------------------|---------------------|-----------------------
                                   |                     |
    verts_packed_to_mesh_idx       |  size = (sum(V_n))  |   tensor([
                                   |                     |     0, 0, 0, 1, 1, 1,
                                   |                     |     1, 2, 2, 2, 2, 2
                                   |                     |   )]
                                   |                     |   size = (12)
                                   |                     |
    mesh_to_verts_packed_first_idx |  size = (N)         |   tensor([0, 3, 7])
                                   |                     |   size = (3)
                                   |                     |
    num_verts_per_mesh             |  size = (N)         |   tensor([3, 4, 5])
                                   |                     |   size = (3)
                                   |                     |
    faces_packed_to_mesh_idx       |  size = (sum(F_n))  |   tensor([
                                   |                     |     0, 1, 1, 2, 2, 2,
                                   |                     |     2, 2, 2, 2
                                   |                     |   )]
                                   |                     |   size = (10)
                                   |                     |
    mesh_to_faces_packed_first_idx |  size = (N)         |   tensor([0, 1, 3])
                                   |                     |   size = (3)
                                   |                     |
    num_faces_per_mesh             |  size = (N)         |   tensor([1, 2, 7])
                                   |                     |   size = (3)
                                   |                     |
    verts_padded_to_packed_idx     |  size = (sum(V_n))  |  tensor([
                                   |                     |     0, 1, 2, 5, 6, 7,
                                   |                     |     8, 10, 11, 12, 13,
                                   |                     |     14
                                   |                     |  )]
                                   |                     |  size = (12)
    -----------------------------------------------------------------------------
    # SPHINX IGNORE

    From the faces, edges are computed and have packed and padded
    representations with auxiliary variables.

    E_n = [[E_1], ... , [E_N]]
    where E_1, ... , E_N are the number of unique edges in each mesh.
    Total number of unique edges = sum(E_n)

    # SPHINX IGNORE
    Name                           |   Size                  | Example from above
    -------------------------------|-------------------------|----------------------
                                   |                         |
    edges_packed                   | size = (sum(E_n), 2)    |  tensor([
                                   |                         |     [0, 1],
                                   |                         |     [0, 2],
                                   |                         |     [1, 2],
                                   |                         |       ...
                                   |                         |     [10, 11],
                                   |                         |   )]
                                   |                         |   size = (18, 2)
                                   |                         |
    num_edges_per_mesh             | size = (N)              |  tensor([3, 5, 10])
                                   |                         |  size = (3)
                                   |                         |
    edges_packed_to_mesh_idx       | size = (sum(E_n))       |  tensor([
                                   |                         |    0, 0, 0,
                                   |                         |     . . .
                                   |                         |    2, 2, 2
                                   |                         |   ])
                                   |                         |   size = (18)
                                   |                         |
    faces_packed_to_edges_packed   | size = (sum(F_n), 3)    |  tensor([
                                   |                         |    [2,   1,  0],
                                   |                         |    [5,   4,  3],
                                   |                         |       .  .  .
                                   |                         |    [12, 14, 16],
                                   |                         |   ])
                                   |                         |   size = (10, 3)
                                   |                         |
    mesh_to_edges_packed_first_idx | size = (N)              |  tensor([0, 3, 8])
                                   |                         |  size = (3)
    ----------------------------------------------------------------------------
    # SPHINX IGNORE
    """

    _INTERNAL_TENSORS = [
        "_verts_packed",
        "_verts_packed_to_mesh_idx",
        "_mesh_to_verts_packed_first_idx",
        "_verts_padded",
        "_num_verts_per_mesh",
        "_faces_packed",
        "_faces_packed_to_mesh_idx",
        "_mesh_to_faces_packed_first_idx",
        "_faces_padded",
        "_faces_areas_packed",
        "_verts_normals_packed",
        "_num_faces_per_mesh",
        "_edges_packed",
        "_edges_packed_to_mesh_idx",
        "_mesh_to_edges_packed_first_idx",
        "_faces_packed_to_edges_packed",
        "_num_edges_per_mesh",
        "_verts_padded_to_packed_idx",
        "valid",
        "equisized",
    ]

    def __init__(
        self,
        verts,
        faces,
        textures=None,
        *,
        verts_normals=None,
    ) -> None:
        """
        Args:
            verts:
                Can be either

                - List where each element is a tensor of shape (num_verts, 3)
                  containing the (x, y, z) coordinates of each vertex.
                - Padded float tensor with shape (num_meshes, max_num_verts, 3).
                  Meshes should be padded with fill value of 0 so they all have
                  the same number of vertices.
            faces:
                Can be either

                - List where each element is a tensor of shape (num_faces, 3)
                  containing the indices of the 3 vertices in the corresponding
                  mesh in verts which form the triangular face.
                - Padded long tensor of shape (num_meshes, max_num_faces, 3).
                  Meshes should be padded with fill value of -1 so they have
                  the same number of faces.
            textures: Optional instance of the Textures class with mesh
                texture properties.
            verts_normals:
                Optional. Can be either

                - List where each element is a tensor of shape (num_verts, 3)
                  containing the normals of each vertex.
                - Padded float tensor with shape (num_meshes, max_num_verts, 3).
                  They should be padded with fill value of 0 so they all have
                  the same number of vertices.
                Note that modifying the mesh later, e.g. with offset_verts_,
                can cause these normals to be forgotten and normals to be recalculated
                based on the new vertex positions.

        Refer to comments above for descriptions of List and Padded representations.
        """
        self.device = torch.device("cpu")
        if textures is not None and not hasattr(textures, "sample_textures"):
            msg = "Expected textures to be an instance of type TexturesBase; got %r"
            raise ValueError(msg % type(textures))

        self.textures = textures

        # Indicates whether the meshes in the list/batch have the same number
        # of faces and vertices.
        self.equisized = False

        # Boolean indicator for each mesh in the batch
        # True if mesh has non zero number of verts and face, False otherwise.
        self.valid = None

        self._N = 0  # batch size (number of meshes)
        self._V = 0  # (max) number of vertices per mesh
        self._F = 0  # (max) number of faces per mesh

        # List of Tensors of verts and faces.
        self._verts_list = None
        self._faces_list = None

        # Packed representation for verts.
        self._verts_packed = None  # (sum(V_n), 3)
        self._verts_packed_to_mesh_idx = None  # sum(V_n)

        # Index to convert verts from flattened padded to packed
        self._verts_padded_to_packed_idx = None  # N * max_V

        # Index of each mesh's first vert in the packed verts.
        # Assumes packing is sequential.
        self._mesh_to_verts_packed_first_idx = None  # N

        # Packed representation for faces.
        self._faces_packed = None  # (sum(F_n), 3)
        self._faces_packed_to_mesh_idx = None  # sum(F_n)

        # Index of each mesh's first face in packed faces.
        # Assumes packing is sequential.
        self._mesh_to_faces_packed_first_idx = None  # N

        # Packed representation of edges sorted by index of the first vertex
        # in the edge. Edges can be shared between faces in a mesh.
        self._edges_packed = None  # (sum(E_n), 2)

        # Map from packed edges to corresponding mesh index.
        self._edges_packed_to_mesh_idx = None  # sum(E_n)
        self._num_edges_per_mesh = None  # N
        self._mesh_to_edges_packed_first_idx = None  # N

        # Map from packed faces to packed edges. This represents the index of
        # the edge opposite the vertex for each vertex in the face. E.g.
        #
        #         v0
        #         /\
        #        /  \
        #    e1 /    \ e2
        #      /      \
        #     /________\
        #   v2    e0   v1
        #
        # Face (v0, v1, v2) => Edges (e0, e1, e2)
        self._faces_packed_to_edges_packed = None  # (sum(F_n), 3)

        # Padded representation of verts.
        self._verts_padded = None  # (N, max(V_n), 3)
        self._num_verts_per_mesh = None  # N

        # Padded representation of faces.
        self._faces_padded = None  # (N, max(F_n), 3)
        self._num_faces_per_mesh = None  # N

        # Face areas
        self._faces_areas_packed = None

        # Normals
        self._verts_normals_packed = None

        # Identify type of verts and faces.
        if isinstance(verts, list) and isinstance(faces, list):
            self._verts_list = verts
            self._faces_list = [f[f.gt(-1).all(1)].to(torch.int64) if len(f) > 0 else f for f in faces]
            self._N = len(self._verts_list)
            self.valid = torch.zeros((self._N,), dtype=torch.bool, device=self.device)
            if self._N > 0:
                self.device = self._verts_list[0].device
                if not (all(v.device == self.device for v in verts) and all(f.device == self.device for f in faces)):
                    raise ValueError("All Verts and Faces tensors should be on same device.")
                self._num_verts_per_mesh = torch.tensor([len(v) for v in self._verts_list], device=self.device)
                self._V = int(self._num_verts_per_mesh.max())
                self._num_faces_per_mesh = torch.tensor([len(f) for f in self._faces_list], device=self.device)
                self._F = int(self._num_faces_per_mesh.max())
                self.valid = torch.tensor(
                    [len(v) > 0 and len(f) > 0 for (v, f) in zip(self._verts_list, self._faces_list)],
                    dtype=torch.bool,
                    device=self.device,
                )
                if (len(self._num_verts_per_mesh.unique()) == 1) and (len(self._num_faces_per_mesh.unique()) == 1):
                    self.equisized = True

        elif torch.is_tensor(verts) and torch.is_tensor(faces):
            if verts.size(2) != 3 or faces.size(2) != 3:
                raise ValueError("Verts or Faces tensors have incorrect dimensions.")
            self._verts_padded = verts
            self._faces_padded = faces.to(torch.int64)
            self._N = self._verts_padded.shape[0]
            self._V = self._verts_padded.shape[1]

            if verts.device != faces.device:
                msg = "Verts and Faces tensors should be on same device. \n Got {} and {}."
                raise ValueError(msg.format(verts.device, faces.device))

            self.device = self._verts_padded.device
            self.valid = torch.zeros((self._N,), dtype=torch.bool, device=self.device)
            if self._N > 0:
                # Check that padded faces - which have value -1 - are at the
                # end of the tensors
                faces_not_padded = self._faces_padded.gt(-1).all(2)
                self._num_faces_per_mesh = faces_not_padded.sum(1)
                if (faces_not_padded[:, :-1] < faces_not_padded[:, 1:]).any():
                    raise ValueError("Padding of faces must be at the end")

                # NOTE that we don't check for the ordering of padded verts
                # as long as the faces index correspond to the right vertices.

                self.valid = self._num_faces_per_mesh > 0
                self._F = int(self._num_faces_per_mesh.max())
                if len(self._num_faces_per_mesh.unique()) == 1:
                    self.equisized = True

                self._num_verts_per_mesh = torch.full(
                    size=(self._N,),
                    fill_value=self._V,
                    dtype=torch.int64,
                    device=self.device,
                )

        else:
            raise ValueError(
                "Verts and Faces must be either a list or a tensor with \
                    shape (batch_size, N, 3) where N is either the maximum \
                       number of verts or faces respectively."
            )

        if self.isempty():
            self._num_verts_per_mesh = torch.zeros((0,), dtype=torch.int64, device=self.device)
            self._num_faces_per_mesh = torch.zeros((0,), dtype=torch.int64, device=self.device)

        # Set the num verts/faces on the textures if present.
        if textures is not None:
            shape_ok = self.textures.check_shapes(self._N, self._V, self._F)
            if not shape_ok:
                msg = "Textures do not match the dimensions of Meshes."
                raise ValueError(msg)

            self.textures._num_faces_per_mesh = self._num_faces_per_mesh.tolist()
            self.textures._num_verts_per_mesh = self._num_verts_per_mesh.tolist()
            self.textures.valid = self.valid

        if verts_normals is not None:
            self._set_verts_normals(verts_normals)

    def _set_verts_normals(self, verts_normals) -> None:
        if isinstance(verts_normals, list):
            if len(verts_normals) != self._N:
                raise ValueError("Invalid verts_normals input")

            for item, n_verts in zip(verts_normals, self._num_verts_per_mesh):
                if (
                    not isinstance(item, torch.Tensor)
                    or item.ndim != 2
                    or item.shape[1] != 3
                    or item.shape[0] != n_verts
                ):
                    raise ValueError("Invalid verts_normals input")
            self._verts_normals_packed = torch.cat(verts_normals, 0)
        elif torch.is_tensor(verts_normals):
            if verts_normals.ndim != 3 or verts_normals.size(2) != 3 or verts_normals.size(0) != self._N:
                raise ValueError("Vertex normals tensor has incorrect dimensions.")
            self._verts_normals_packed = padded_to_packed(verts_normals, split_size=self._num_verts_per_mesh.tolist())
        else:
            raise ValueError("verts_normals must be a list or tensor")

    def __len__(self) -> int:
        return self._N

    def __getitem__(self, index: Union[int, List[int], slice, torch.BoolTensor, torch.LongTensor]) -> "Meshes":
        """
        Args:
            index: Specifying the index of the mesh to retrieve.
                Can be an int, slice, list of ints or a boolean tensor.

        Returns:
            Meshes object with selected meshes. The mesh tensors are not cloned.
        """
        if isinstance(index, (int, slice)):
            verts = self.verts_list()[index]
            faces = self.faces_list()[index]
        elif isinstance(index, list):
            verts = [self.verts_list()[i] for i in index]
            faces = [self.faces_list()[i] for i in index]
        elif isinstance(index, torch.Tensor):
            if index.dim() != 1 or index.dtype.is_floating_point:
                raise IndexError(index)
            # NOTE consider converting index to cpu for efficiency
            if index.dtype == torch.bool:
                # advanced indexing on a single dimension
                index = index.nonzero()
                index = index.squeeze(1) if index.numel() > 0 else index
                index = index.tolist()
            verts = [self.verts_list()[i] for i in index]
            faces = [self.faces_list()[i] for i in index]
        else:
            raise IndexError(index)

        textures = None if self.textures is None else self.textures[index]

        if torch.is_tensor(verts) and torch.is_tensor(faces):
            return self.__class__(verts=[verts], faces=[faces], textures=textures)
        elif isinstance(verts, list) and isinstance(faces, list):
            return self.__class__(verts=verts, faces=faces, textures=textures)
        else:
            raise ValueError("(verts, faces) not defined correctly")

    def isempty(self) -> bool:
        """
        Checks whether any mesh is valid.

        Returns:
            bool indicating whether there is any data.
        """
        return self._N == 0 or self.valid.eq(False).all()

    def verts_list(self):
        """
        Get the list representation of the vertices.

        Returns:
            list of tensors of vertices of shape (V_n, 3).
        """
        if self._verts_list is None:
            assert self._verts_padded is not None, "verts_padded is required to compute verts_list."
            self._verts_list = padded_to_list(self._verts_padded, self.num_verts_per_mesh().tolist())
        return self._verts_list

    def faces_list(self):
        """
        Get the list representation of the faces.

        Returns:
            list of tensors of faces of shape (F_n, 3).
        """
        if self._faces_list is None:
            assert self._faces_padded is not None, "faces_padded is required to compute faces_list."
            self._faces_list = padded_to_list(self._faces_padded, self.num_faces_per_mesh().tolist())
        return self._faces_list

    def verts_packed(self):
        """
        Get the packed representation of the vertices.

        Returns:
            tensor of vertices of shape (sum(V_n), 3).
        """
        self._compute_packed()
        return self._verts_packed

    def num_verts_per_mesh(self):
        """
        Return a 1D tensor x with length equal to the number of meshes giving
        the number of vertices in each mesh.

        Returns:
            1D tensor of sizes.
        """
        return self._num_verts_per_mesh

    def faces_packed(self):
        """
        Get the packed representation of the faces.
        Faces are given by the indices of the three vertices in verts_packed.

        Returns:
            tensor of faces of shape (sum(F_n), 3).
        """
        self._compute_packed()
        return self._faces_packed

    def num_faces_per_mesh(self):
        """
        Return a 1D tensor x with length equal to the number of meshes giving
        the number of faces in each mesh.

        Returns:
            1D tensor of sizes.
        """
        return self._num_faces_per_mesh

    def has_verts_normals(self) -> bool:
        """
        Check whether vertex normals are already present.
        """
        return self._verts_normals_packed is not None

    def verts_normals_packed(self):
        """
        Get the packed representation of the vertex normals.

        Returns:
            tensor of normals of shape (sum(V_n), 3).
        """
        self._compute_vertex_normals()
        return self._verts_normals_packed

    def verts_normals_list(self):
        """
        Get the list representation of the vertex normals.

        Returns:
            list of tensors of normals of shape (V_n, 3).
        """
        if self.isempty():
            return [torch.empty((0, 3), dtype=torch.float32, device=self.device)] * self._N
        verts_normals_packed = self.verts_normals_packed()
        split_size = self.num_verts_per_mesh().tolist()
        return packed_to_list(verts_normals_packed, split_size)

    def _compute_vertex_normals(self, refresh: bool = False):
        """Computes the packed version of vertex normals from the packed verts
        and faces. This assumes verts are shared between faces. The normal for
        a vertex is computed as the sum of the normals of all the faces it is
        part of weighed by the face areas.

        Args:
            refresh: Set to True to force recomputation of vertex normals.
                Default: False.
        """
        if not (refresh or any(v is None for v in [self._verts_normals_packed])):
            return

        if self.isempty():
            self._verts_normals_packed = torch.zeros((self._N, 3), dtype=torch.int64, device=self.device)
        else:
            faces_packed = self.faces_packed()
            verts_packed = self.verts_packed()
            verts_normals = torch.zeros_like(verts_packed)
            vertices_faces = verts_packed[faces_packed]

            faces_normals = torch.cross(
                vertices_faces[:, 2] - vertices_faces[:, 1],
                vertices_faces[:, 0] - vertices_faces[:, 1],
                dim=1,
            )

            # NOTE: this is already applying the area weighting as the magnitude
            # of the cross product is 2 x area of the triangle.
            verts_normals = verts_normals.index_add(0, faces_packed[:, 0], faces_normals)
            verts_normals = verts_normals.index_add(0, faces_packed[:, 1], faces_normals)
            verts_normals = verts_normals.index_add(0, faces_packed[:, 2], faces_normals)

            self._verts_normals_packed = torch.nn.functional.normalize(verts_normals, eps=1e-6, dim=1)

    # TODO(nikhilar) Improve performance of _compute_packed.
    def _compute_packed(self, refresh: bool = False):
        """
        Computes the packed version of the meshes from verts_list and faces_list
        and sets the values of auxiliary tensors.

        Args:
            refresh: Set to True to force recomputation of packed representations.
                Default: False.
        """

        if not (
            refresh
            or any(
                v is None
                for v in [
                    self._verts_packed,
                    self._verts_packed_to_mesh_idx,
                    self._mesh_to_verts_packed_first_idx,
                    self._faces_packed,
                    self._faces_packed_to_mesh_idx,
                    self._mesh_to_faces_packed_first_idx,
                ]
            )
        ):
            return

        # Packed can be calculated from padded or list, so can call the
        # accessor function for verts_list and faces_list.
        verts_list = self.verts_list()
        faces_list = self.faces_list()
        if self.isempty():
            self._verts_packed = torch.zeros((0, 3), dtype=torch.float32, device=self.device)
            self._verts_packed_to_mesh_idx = torch.zeros((0,), dtype=torch.int64, device=self.device)
            self._mesh_to_verts_packed_first_idx = torch.zeros((0,), dtype=torch.int64, device=self.device)
            self._num_verts_per_mesh = torch.zeros((0,), dtype=torch.int64, device=self.device)
            self._faces_packed = -(torch.ones((0, 3), dtype=torch.int64, device=self.device))
            self._faces_packed_to_mesh_idx = torch.zeros((0,), dtype=torch.int64, device=self.device)
            self._mesh_to_faces_packed_first_idx = torch.zeros((0,), dtype=torch.int64, device=self.device)
            self._num_faces_per_mesh = torch.zeros((0,), dtype=torch.int64, device=self.device)
            return

        verts_list_to_packed = list_to_packed(verts_list)
        self._verts_packed = verts_list_to_packed[0]
        if not torch.allclose(self.num_verts_per_mesh(), verts_list_to_packed[1]):
            raise ValueError("The number of verts per mesh should be consistent.")
        self._mesh_to_verts_packed_first_idx = verts_list_to_packed[2]
        self._verts_packed_to_mesh_idx = verts_list_to_packed[3]

        faces_list_to_packed = list_to_packed(faces_list)
        faces_packed = faces_list_to_packed[0]
        if not torch.allclose(self.num_faces_per_mesh(), faces_list_to_packed[1]):
            raise ValueError("The number of faces per mesh should be consistent.")
        self._mesh_to_faces_packed_first_idx = faces_list_to_packed[2]
        self._faces_packed_to_mesh_idx = faces_list_to_packed[3]

        faces_packed_offset = self._mesh_to_verts_packed_first_idx[self._faces_packed_to_mesh_idx]
        self._faces_packed = faces_packed + faces_packed_offset.view(-1, 1)

    def detach(self):
        """
        Detach Meshes object. All internal tensors are detached individually.

        Returns:
            new Meshes object.
        """
        verts_list = self.verts_list()
        faces_list = self.faces_list()
        new_verts_list = [v.detach() for v in verts_list]
        new_faces_list = [f.detach() for f in faces_list]
        other = self.__class__(verts=new_verts_list, faces=new_faces_list)
        for k in self._INTERNAL_TENSORS:
            v = getattr(self, k)
            if torch.is_tensor(v):
                setattr(other, k, v.detach())

        # Textures is not a tensor but has a detach method
        if self.textures is not None:
            other.textures = self.textures.detach()
        return other

    def to(self, device: Device, copy: bool = False):
        """
        Match functionality of torch.Tensor.to()
        If copy = True or the self Tensor is on a different device, the
        returned tensor is a copy of self with the desired torch.device.
        If copy = False and the self Tensor already has the correct torch.device,
        then self is returned.

        Args:
            device: Device (as str or torch.device) for the new tensor.
            copy: Boolean indicator whether or not to clone self. Default False.

        Returns:
            Meshes object.
        """
        device_ = make_device(device)
        if not copy and self.device == device_:
            return self

        other = self.clone()
        if self.device == device_:
            return other

        other.device = device_
        if other._N > 0:
            other._verts_list = [v.to(device_) for v in other._verts_list]
            other._faces_list = [f.to(device_) for f in other._faces_list]
        for k in self._INTERNAL_TENSORS:
            v = getattr(self, k)
            if torch.is_tensor(v):
                setattr(other, k, v.to(device_))
        if self.textures is not None:
            other.textures = other.textures.to(device_)
        return other

    def cpu(self):
        return self.to("cpu")

    def cuda(self):
        return self.to("cuda")


def join_meshes_as_batch(meshes: List[Meshes], include_textures: bool = True) -> Meshes:
    """
    Merge multiple Meshes objects, i.e. concatenate the meshes objects. They
    must all be on the same device. If include_textures is true, they must all
    be compatible, either all or none having textures, and all the Textures
    objects being the same type. If include_textures is False, textures are
    ignored.

    If the textures are TexturesAtlas then being the same type includes having
    the same resolution. If they are TexturesUV then it includes having the same
    align_corners and padding_mode.

    Args:
        meshes: list of meshes.
        include_textures: (bool) whether to try to join the textures.

    Returns:
        new Meshes object containing all the meshes from all the inputs.
    """
    if isinstance(meshes, Meshes):
        # Meshes objects can be iterated and produce single Meshes. We avoid
        # letting join_meshes_as_batch(mesh1, mesh2) silently do the wrong thing.
        raise ValueError("Wrong first argument to join_meshes_as_batch.")
    verts = [v for mesh in meshes for v in mesh.verts_list()]
    faces = [f for mesh in meshes for f in mesh.faces_list()]
    if len(meshes) == 0 or not include_textures:
        return Meshes(verts=verts, faces=faces)

    if meshes[0].textures is None:
        if any(mesh.textures is not None for mesh in meshes):
            raise ValueError("Inconsistent textures in join_meshes_as_batch.")
        return Meshes(verts=verts, faces=faces)

    if any(mesh.textures is None for mesh in meshes):
        raise ValueError("Inconsistent textures in join_meshes_as_batch.")

    # Now we know there are multiple meshes and they have textures to merge.
    all_textures = [mesh.textures for mesh in meshes]
    first = all_textures[0]
    tex_types_same = all(type(tex) == type(first) for tex in all_textures)

    if not tex_types_same:
        raise ValueError("All meshes in the batch must have the same type of texture.")

    tex = first.join_batch(all_textures[1:])
    return Meshes(verts=verts, faces=faces, textures=tex)


def join_meshes_as_scene(meshes: Union[Meshes, List[Meshes]], include_textures: bool = True) -> Meshes:
    """
    Joins a batch of meshes in the form of a Meshes object or a list of Meshes
    objects as a single mesh. If the input is a list, the Meshes objects in the
    list must all be on the same device. Unless include_textures is False, the
    meshes must all have the same type of texture or must all not have textures.

    If textures are included, then the textures are joined as a single scene in
    addition to the meshes. For this, texture types have an appropriate method
    called join_scene which joins mesh textures into a single texture.
    If the textures are TexturesAtlas then they must have the same resolution.
    If they are TexturesUV then they must have the same align_corners and
    padding_mode. Values in verts_uvs outside [0, 1] will not
    be respected.

    Args:
        meshes: Meshes object that contains a batch of meshes, or a list of
                    Meshes objects.
        include_textures: (bool) whether to try to join the textures.

    Returns:
        new Meshes object containing a single mesh
    """
    if not isinstance(include_textures, (bool, int)):
        # We want to avoid letting join_meshes_as_scene(mesh1, mesh2) silently
        # do the wrong thing.
        raise ValueError(f"include_textures argument cannot be {type(include_textures)}")
    if isinstance(meshes, List):
        meshes = join_meshes_as_batch(meshes, include_textures=include_textures)

    if len(meshes) == 1:
        return meshes
    verts = meshes.verts_packed()  # (sum(V_n), 3)
    # Offset automatically done by faces_packed
    faces = meshes.faces_packed()  # (sum(F_n), 3)
    textures = None

    if include_textures and meshes.textures is not None:
        textures = meshes.textures.join_scene()

    mesh = Meshes(verts=verts.unsqueeze(0), faces=faces.unsqueeze(0), textures=textures)
    return mesh


def make_device(device: Device) -> torch.device:
    """
    Makes an actual torch.device object from the device specified as
    either a string or torch.device object. If the device is `cuda` without
    a specific index, the index of the current device is assigned.

    Args:
        device: Device (as str or torch.device)

    Returns:
        A matching torch.device object
    """
    device = torch.device(device) if isinstance(device, str) else device
    if device.type == "cuda" and device.index is None:
        # If cuda but with no index, then the current cuda device is indicated.
        # In that case, we fix to that device
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
    return device


def padded_to_list(
    x: torch.Tensor,
    split_size: Union[Sequence[int], Sequence[Sequence[int]], None] = None,
):
    r"""
    Transforms a padded tensor of shape (N, S_1, S_2, ..., S_D) into a list
    of N tensors of shape:
    - (Si_1, Si_2, ..., Si_D) where (Si_1, Si_2, ..., Si_D) is specified in split_size(i)
    - or (S_1, S_2, ..., S_D) if split_size is None
    - or (Si_1, S_2, ..., S_D) if split_size(i) is an integer.

    Args:
      x: tensor
      split_size: optional 1D or 2D list/tuple of ints defining the number of
        items for each tensor.

    Returns:
      x_list: a list of tensors sharing the memory with the input.
    """
    x_list = list(x.unbind(0))

    if split_size is None:
        return x_list

    N = len(split_size)
    if x.shape[0] != N:
        raise ValueError("Split size must be of same length as inputs first dimension")

    for i in range(N):
        if isinstance(split_size[i], int):
            x_list[i] = x_list[i][: split_size[i]]
        else:
            slices = tuple(slice(0, s) for s in split_size[i])  # pyre-ignore
            x_list[i] = x_list[i][slices]
    return x_list


def list_to_packed(x: List[torch.Tensor]):
    r"""
    Transforms a list of N tensors each of shape (Mi, K, ...) into a single
    tensor of shape (sum(Mi), K, ...).

    Args:
      x: list of tensors.

    Returns:
        4-element tuple containing

        - **x_packed**: tensor consisting of packed input tensors along the
          1st dimension.
        - **num_items**: tensor of shape N containing Mi for each element in x.
        - **item_packed_first_idx**: tensor of shape N indicating the index of
          the first item belonging to the same element in the original list.
        - **item_packed_to_list_idx**: tensor of shape sum(Mi) containing the
          index of the element in the list the item belongs to.
    """
    N = len(x)
    num_items = torch.zeros(N, dtype=torch.int64, device=x[0].device)
    item_packed_first_idx = torch.zeros(N, dtype=torch.int64, device=x[0].device)
    item_packed_to_list_idx = []
    cur = 0
    for i, y in enumerate(x):
        num = len(y)
        num_items[i] = num
        item_packed_first_idx[i] = cur
        item_packed_to_list_idx.append(torch.full((num,), i, dtype=torch.int64, device=y.device))
        cur += num

    x_packed = torch.cat(x, dim=0)
    item_packed_to_list_idx = torch.cat(item_packed_to_list_idx, dim=0)

    return x_packed, num_items, item_packed_first_idx, item_packed_to_list_idx


def packed_to_list(x: torch.Tensor, split_size: Union[list, int]):
    r"""
    Transforms a tensor of shape (sum(Mi), K, L, ...) to N set of tensors of
    shape (Mi, K, L, ...) where Mi's are defined in split_size

    Args:
    x: tensor
    split_size: list, tuple or int defining the number of items for each tensor
        in the output list.

    Returns:
    x_list: A list of Tensors
    """
    return x.split(split_size, dim=0)


def padded_to_packed(
    x: torch.Tensor,
    split_size: Union[list, tuple, None] = None,
    pad_value: Union[float, int, None] = None,
):
    r"""
    Transforms a padded tensor of shape (N, M, K) into a packed tensor
    of shape:
     - (sum(Mi), K) where (Mi, K) are the dimensions of
        each of the tensors in the batch and Mi is specified by split_size(i)
     - (N*M, K) if split_size is None

    Support only for 3-dimensional input tensor and 1-dimensional split size.

    Args:
      x: tensor
      split_size: list, tuple or int defining the number of items for each tensor
        in the output list.
      pad_value: optional value to use to filter the padded values in the input
        tensor.

    Only one of split_size or pad_value should be provided, or both can be None.

    Returns:
      x_packed: a packed tensor.
    """
    if x.ndim != 3:
        raise ValueError("Supports only 3-dimensional input tensors")

    N, M, D = x.shape

    if split_size is not None and pad_value is not None:
        raise ValueError("Only one of split_size or pad_value should be provided.")

    x_packed = x.reshape(-1, D)  # flatten padded

    if pad_value is None and split_size is None:
        return x_packed

    # Convert to packed using pad value
    if pad_value is not None:
        mask = x_packed.ne(pad_value).any(-1)
        x_packed = x_packed[mask]
        return x_packed

    # Convert to packed using split sizes
    # pyre-fixme[6]: Expected `Sized` for 1st param but got `Union[None,
    #  List[typing.Any], typing.Tuple[typing.Any, ...]]`.
    N = len(split_size)
    if x.shape[0] != N:
        raise ValueError("Split size must be of same length as inputs first dimension")

    # pyre-fixme[16]: `None` has no attribute `__iter__`.
    if not all(isinstance(i, int) for i in split_size):
        raise ValueError(
            "Support only 1-dimensional unbinded tensor. \
                Split size for more dimensions provided"
        )

    padded_to_packed_idx = torch.cat(
        [
            torch.arange(v, dtype=torch.int64, device=x.device) + i * M
            # pyre-fixme[6]: Expected `Iterable[Variable[_T]]` for 1st param but got
            #  `Union[None, List[typing.Any], typing.Tuple[typing.Any, ...]]`.
            for (i, v) in enumerate(split_size)
        ],
        dim=0,
    )

    return x_packed[padded_to_packed_idx]
