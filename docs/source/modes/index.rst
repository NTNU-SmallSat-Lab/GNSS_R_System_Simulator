The two reflection modes
========================

The simulator can look at the sea in two different ways. They use the same satellites and the same
link-budget equation, but the signal takes a different path, the sea returns the signal by different
physics (a forward specular reflection and glistening zone versus small-ripple Bragg backscatter), and a ship
shows up in a different way.

- **Specular (forward-scatter) mode** -- the signal reflects from the sea surface in the forward (specular) direction, on its way
  down from the GNSS satellite to the LEO receiver. This is the classic GNSS-reflectometry geometry.
- **Backscatter mode** -- the signal comes almost straight back toward where it came from, the way an
  ordinary radar sees a target.

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - Specular reflection
     - Backscatter
   * - .. image:: ../images/specular.png
          :width: 100%
     - .. image:: ../images/Backscatter.png
          :width: 100%

The mode is chosen by one INI key, ``[REFLECTION].reflection_type``. Each page below explains the geometry,
the sea return, how a ship is detected, and which code modules do the work.

Side by side
------------

.. list-table::
   :widths: 22 39 39
   :header-rows: 1

   * - Aspect
     - Specular (forward)
     - Backscatter
   * - Signal path
     - Transmitter (GNSS) -> forward-scatter on the sea -> receiver (GNSS-R).
     - Transmitter (GNSS) -> backscatter on the sea -> receiver (GNSS-R).
   * - Surface point
     - The specular reflection point (:doc:`../Documentation/object_definitions/reflection`).
     - The point where the transmitter-receiver line meets the surface
       (:doc:`../Documentation/object_definitions/backscatter_point`).
   * - Sea return
     - Bright: the specular reflection plus the surrounding glistening zone.
     - Weak: small-ripple (Bragg) scatter.
   * - Detection metric
     - Modelled loss of diffuse ocean return over the ship waterline area (shadow metric).
     - Deterministic signal-to-clutter and signal-to-clutter-plus-noise metrics.
   * - INI setting
     - ``reflection_type = "specular"``
     - ``reflection_type = "backscatter"``

.. toctree::
   :maxdepth: 1
   :caption: The two modes

   specular
   backscatter
